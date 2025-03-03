from scene.multipart_models import *

class OptimOMP(MPArtModelBasic):
    """
    Optimization only on motion parameters.
    """
    def setup_args_extra(self):
        # self.opt.iterations = 9000
        self.opt.iterations = 15000
        self.opt.warmup_until_iter = 1000
        self.opt.cd_from_iter = self.opt.warmup_until_iter + 1
        self.opt.cd_until_iter = 9000

        self.opt.cd_weight = None
        # self.opt.cd_weight = 1.0
        # self.opt.depth_weight = None
        # self.opt.depth_weight = None
        self.opt.depth_weight = 0.1
        # self.opt.rgb_weight = 0
        self.opt.rgb_weight = 0.1

        self.opt.mask_thresh = .85
        self.opt.trace_r_thresh = 1 + 2 * math.cos(5 / 180 * math.pi)

    def __init__(self, gaussians: GaussianModel, num_movable: int):
        super().__init__(gaussians, num_movable)
        self.mpp = None
        self.ppp = None
        self.original_xyz = self.gaussians.get_xyz.clone().detach()
        self.original_rotation = self.gaussians.get_rotation.clone().detach()
        self.original_opacity = self.gaussians.get_opacity.clone().detach()
        self.is_revolute = np.array([True for _ in range(self.num_movable)])
        self.setup_args_extra()

    @override
    def set_dataset(self, source_path: str, model_path: str, evaluate=True):
        self.dataset.eval = evaluate
        self.dataset.source_path = source_path
        self.dataset.model_path = model_path

        mpp = np.load(os.path.join(model_path, 'mpp_init.npy'))
        ppp = np.load(os.path.join(model_path, 'ppp_init.npy'))
        self.mpp = torch.tensor(mpp, device='cuda')
        self.ppp = torch.tensor(ppp, device='cuda')

    @property
    def part_indices(self):
        return torch.argmax(self.ppp, dim=1)

    @property
    def mask(self):
        return self.mpp > self.opt.mask_thresh

    @override
    def deform(self):
        t = self.get_t
        r = self.get_r
        for k in range(self.num_movable):
            msk = self.mask & (self.part_indices == k)
            r_inv_quat = mat2quat(r[k].transpose(1, 0))
            self.gaussians.get_xyz[msk] = torch.matmul(self.original_xyz[msk], r[k]) + t[k]
            self.gaussians.get_rotation_raw[msk] = quat_mult(r_inv_quat, self.original_rotation[msk])
        return self.gaussians

    def _show_losses(self, iteration: int, losses: dict):
        if iteration in [1000, 5000, 9000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
            )

        if iteration not in [2, 20, 50, 200, 500, 1000, 3000, 5000, 7000, 9000, 15000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        for k in range(self.num_movable):
            print(f't{k}:', self.get_t[k].detach().cpu().numpy())
            print(f'r{k}:', self.get_r[k].detach().cpu().numpy())
        print()

    def _eval_losses(self, iteration: int, render_pkg, viewpoint_cam, gaussians, gt_gaussians=None):
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'rgb': eval_img_loss(render_pkg['render'], gt_image, self.opt),
            'd': None,
            'cd': None,
        }
        loss = self.opt.rgb_weight * losses['rgb']
        requires_cd = (self.opt.cd_from_iter <= iteration <= self.opt.cd_until_iter)
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            losses['cd'] = eval_cd_loss_sd(gaussians, gt_gaussians)
            loss += self.opt.cd_weight * losses['cd']
        if (self.opt.depth_weight is not None) and (viewpoint_cam.image_depth is not None):
            gt_depth = viewpoint_cam.image_depth.cuda()
            losses['d'] = eval_depth_loss(render_pkg['depth'], gt_depth)
            loss += self.opt.depth_weight * losses['d']
        return loss, losses

    @override
    def train(self, gt_gaussians=None):
        _ = prepare_output_and_logger(self.dataset)
        iterations = self.opt.iterations
        self.training_setup(self.opt)
        bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(iterations), desc="Training progress")
        for i in range(1, iterations + 1):
            self.deform()

            # Pick a random Camera
            viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
            loss, losses = self._eval_losses(i, render_pkg, viewpoint_cam, self.gaussians, gt_gaussians)
            loss.backward()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                if i < iterations:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
                    self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach()

                if i == self.opt.warmup_until_iter:
                    print()
                    for k in range(self.num_movable):
                        self.is_revolute[k] = torch.trace(self.get_r[k]) < self.opt.trace_r_thresh
                        print(f'Detected part{k} is ' + ('*REVOLUTE*.' if self.is_revolute[k] else '*PRISMATIC*.'))
                        if self.is_revolute[k]:
                            continue
                        # self._column_vec1[k] = nn.Parameter(
                        #     torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        # )
                        # self._column_vec2[k] = nn.Parameter(
                        #     torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        # )
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r


class MPArtModelII(MPArtModel):
    def __init__(self, gaussians: GaussianModel, num_movable: int):
        super().__init__(gaussians, num_movable)

    @override
    def set_dataset(self, source_path: str, model_path: str, evaluate=True):
        self.dataset.eval = evaluate
        self.dataset.source_path = source_path
        self.dataset.model_path = model_path

        mpp = torch.tensor(np.load(os.path.join(model_path, 'mpp_init.npy')), device='cuda')
        ppp = torch.tensor(np.load(os.path.join(model_path, 'ppp_init.npy')), device='cuda')
        self.set_init_probabilities(prob=mpp, ppp=ppp)

    @override
    def deform(self):
        t = self.get_t
        r = self.get_r
        prob = self.get_prob.unsqueeze(-1)
        ppp = self.get_ppp.unsqueeze(-1)

        self.gaussians.get_xyz[:] = self.original_xyz * (1 - prob)
        self.gaussians.get_rotation_raw[:] = self.original_rotation * (1 - prob)
        for k in range(self.num_movable):
            r_inv_quat = mat2quat(r[k].transpose(1, 0))
            self.gaussians.get_xyz[:] += (torch.matmul(self.original_xyz, r[k]) + t[k]) * prob * ppp[:, k]
            self.gaussians.get_rotation_raw[:] += quat_mult(r_inv_quat, self.original_rotation) * prob * ppp[:, k]
        return self.gaussians

    @override
    def training_setup(self, training_args):
        l = [
            {'params': self._column_vec1, 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': self._column_vec2, 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': self._t, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': [self._prob], 'lr': training_args.prob_lr, "name": "prob"},
            {'params': [self._ppp], 'lr': training_args.prob_lr, "name": "ppp"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @override
    def _eval_losses(self, render_pkg, viewpoint_cam, gaussians, gt_gaussians=None, requires_cd=False):
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'im': eval_img_loss(render_pkg['render'], gt_image, self.opt),
            'cd': None,
            'd': None,
        }
        loss = losses['im']
        if (self.opt.depth_weight is not None) and (viewpoint_cam.image_depth is not None):
            gt_depth = viewpoint_cam.image_depth.cuda()
            losses['d'] = eval_depth_loss(render_pkg['depth'], gt_depth)
            loss += self.opt.depth_weight * losses['d']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            losses['cd'] = eval_cd_loss_sd(gaussians, gt_gaussians)
            loss += self.opt.cd_weight * losses['cd']
        return loss, losses


