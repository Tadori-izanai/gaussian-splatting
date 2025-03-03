import os
import math
import torch
from torch import nn
from tqdm import tqdm
from random import randint

from typing_extensions import override

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene.articulation_model import ArticulationModelBasic
from scene import Scene, BWScenes
from utils.general_utils import quat_mult, mat2quat, inverse_sigmoid
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses, eval_cd_loss_sd, eval_depth_loss
from train import prepare_output_and_logger

class ArticulationModel(ArticulationModelBasic):
    def setup_args_extra(self):
        self.opt.iterations = 15000
        self.opt.warmup_until_iter = 1000
        self.opt.cd_from_iter = self.opt.warmup_until_iter + 1
        self.opt.cd_until_iter = 5000

        # self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.00005
        self.opt.prob_lr = 0.05
        self.opt.cd_weight = 1.0
        self.opt.bce_weight = None

        # self.opt.mask_thresh = .514
        self.opt.mask_thresh = .85
        self.opt.trace_r_thresh = 1 + 2 * math.cos(5 / 180 * math.pi)

        # self.opt.depth_weight = None
        self.opt.depth_weight = 1.0
        self.opt.cd_n_sample = None
        # self.opt.cd_n_sample = 10000

    def __init__(self, gaussians: GaussianModel):
        super().__init__(gaussians)
        self._prob = nn.Parameter(
            torch.zeros(gaussians.size(), dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self.prob_activation = torch.sigmoid
        self.original_opacity = self.gaussians.get_opacity.clone().detach()
        self.gaussians.duplicate()
        self.is_revolute = True
        self.setup_args_extra()

    @property
    def get_prob(self):
        return self.prob_activation(self._prob)

    def set_init_prob(self, prob: torch.tensor, eps=1e-6):
        prob_raw = inverse_sigmoid(torch.clamp(prob, eps, 1 - eps))
        self._prob = nn.Parameter(
            torch.tensor(prob_raw, dtype=torch.float, device='cuda').requires_grad_(False)
        )

    @override
    def deform(self):
        num = self.gaussians.size() // 2
        r = self.get_r
        r_inv_quat = mat2quat(r.transpose(1, 0))
        prob = self.get_prob.unsqueeze(-1)

        self.gaussians.get_xyz[:num] = torch.matmul(self.original_xyz, r) + self.get_t
        self.gaussians.get_rotation_raw[:num] = quat_mult(r_inv_quat, self.original_rotation)
        self.gaussians.get_opacity_raw[:num] = inverse_sigmoid(prob * self.original_opacity)
        self.gaussians.get_opacity_raw[num: 2 * num] = inverse_sigmoid((1 - prob) * self.original_opacity)
        return self.gaussians

    def cancel_se3_grads(self):
        self._t.requires_grad_(False)
        self._column_vec1.requires_grad_(False)
        self._column_vec2.requires_grad_(False)

    @override
    def training_setup(self, training_args):
        print(self.gaussians.spatial_lr_scale)
        l = [
            {'params': [self._column_vec1], 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': [self._column_vec2], 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': [self._t], 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': [self._prob], 'lr': training_args.prob_lr, "name": "prob"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @override
    def _show_losses(self, iteration: int, losses: dict):
        if iteration in [1000, 5000, 9000, 15000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply'),
                prune=False
            )

        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 5000, 9000, 15000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        print('t:', self.get_t.detach().cpu().numpy())
        print('r:', self.get_r.detach().cpu().numpy())
        # print('_column_vec1:', self._column_vec1.detach().cpu().numpy())
        # print('_column_vec2:', self._column_vec2.detach().cpu().numpy())
        # print('prob min: ', torch.min(self._prob), torch.min(self.get_prob))
        # print('prob max: ', torch.max(self._prob), torch.max(self.get_prob))
        print()

    @override
    def _eval_losses(self, render_pkg, viewpoint_cam, gaussians, gt_gaussians=None, requires_cd=False):
        image = render_pkg['render']
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'im': eval_img_loss(image, gt_image, self.opt),
            'cd': None,
            'bce': None,
            'd': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            num = gaussians.size() // 2
            definite_gaussians = GaussianModel(0)
            # if self.is_revolute:
            #     definite_gaussians.get_xyz = torch.cat(
            #         (
            #             gaussians.get_xyz[:num][self.get_prob > self.opt.mask_thresh],
            #             gaussians.get_xyz[num: 2 * num][self.get_prob < (1 - self.opt.mask_thresh)]
            #         ), dim=0
            #     )
            # else:
            #     definite_gaussians.get_xyz = gaussians.get_xyz[:num][self.get_prob > self.opt.mask_thresh]
            definite_gaussians.get_xyz = torch.cat(
                (
                    gaussians.get_xyz[:num][self.get_prob > self.opt.mask_thresh],
                    gaussians.get_xyz[num: 2 * num][self.get_prob < (1 - self.opt.mask_thresh)]
                ), dim=0
            )

            # losses['cd'] = eval_cd_loss(gaussians, gt_gaussians, self.opt.cd_numbers)
            losses['cd'] = eval_cd_loss_sd(definite_gaussians, gt_gaussians, n_samples=self.opt.cd_n_sample)
            loss += self.opt.cd_weight * losses['cd']
        if self.opt.bce_weight is not None:
            losses['bce'] = self._eval_bce_loss()
            loss += self.opt.bce_weight * losses['bce']
        if (self.opt.depth_weight is not None) and (viewpoint_cam.image_depth is not None):
            depth = render_pkg['depth']
            gt_depth = viewpoint_cam.image_depth.cuda()
            losses['d'] = eval_depth_loss(depth, gt_depth)
            loss += self.opt.depth_weight * losses['d']
        return loss, losses

    def _eval_bce_loss(self):
        eps = 1e-5      # cannot be smaller
        p = self.get_prob
        p = torch.clamp(p, min=eps, max=1 - eps)
        return torch.nn.functional.binary_cross_entropy(p, p, reduction='mean')

    @override
    def train(self, gt_gaussians=None):
        _ = prepare_output_and_logger(self.dataset)
        iterations = self.opt.iterations
        bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)
        self.training_setup(self.opt)

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(iterations), desc="Training progress")
        for i in range(1, iterations + 1):
            # Pick a random Camera
            viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()

            self.deform()

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
            requires_cd = (self.opt.cd_from_iter <= i <= self.opt.cd_until_iter)
            loss, losses = self._eval_losses(render_pkg, viewpoint_cam, self.gaussians, gt_gaussians, requires_cd=requires_cd)
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
                    self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach()
                    self._prob[:] = torch.clamp(self._prob, -16, 16)

                if i == self.opt.warmup_until_iter:
                    self.is_revolute = torch.trace(self.get_r) < self.opt.trace_r_thresh
                    print('Detected *REVOLUTE*' if self.is_revolute else 'Detected *PRISMATIC*')
                    if not self.is_revolute:
                        self._column_vec1 = nn.Parameter(
                            torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        )
                        self._column_vec2 = nn.Parameter(
                            torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        )
                    self._prob.requires_grad_(True)
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r
