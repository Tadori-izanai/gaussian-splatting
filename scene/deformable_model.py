#
# Created by lxl.
#

import os
import torch
from torch import nn
from tqdm import tqdm
import math
import copy
import numpy as np

from typing_extensions import override

from pytorch3d.loss import chamfer_distance

from gaussian_renderer import render
from arguments import GroupParams, get_default_args
from scene.gaussian_model import GaussianModel
from scene import BWScenes
from scene.network import DeformNet
from scene.dataset_readers import fetchPly, storePly

from utils.general_utils import get_expon_lr_func, build_rotation, quat_mult, weighted_l2_loss_v2, knn, inverse_sigmoid
from utils.loss_utils import eval_img_loss, eval_depth_loss, eval_rigid_loss, eval_iso_loss
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from train import prepare_output_and_logger

from plyfile import PlyData, PlyElement

class DeformationBasic:
    def setup_args(self):
        self.pipe.convert_SHs_python = False
        self.pipe.compute_cov3D_python = False
        self.pipe.debug = False

        self.dataset.sh_degree = 0
        self.dataset.source_path = None
        self.dataset.model_path = None
        self.dataset.images = "images"
        self.dataset.resolution = -1
        self.dataset.white_background = False
        self.dataset.data_device = "cuda"
        self.dataset.eval = False

        self.opt.iterations = 10_000
        self.opt.lambda_dssim = 0.2

    def __init__(self, gaussians: GaussianModel):
        self.gaussians = gaussians
        self.optimizer = None
        self.dataset = GroupParams()
        self.opt = GroupParams()
        self.pipe = GroupParams()
        self.setup_function()

    def setup_function(self):
        self.gaussians.cancel_grads()
        self.setup_args()

    def set_dataset(self, source_path: str, model_path: str, evaluate=True):
        self.dataset.eval = evaluate
        self.dataset.source_path = source_path
        self.dataset.model_path = model_path

    def deform(self):
        pass

    def training_setup(self, training_args):
        pass

    def train(self):
        pass

class DMCanonical(DeformationBasic):
    def setup_args_extra(self):
        self.opt.densify_grad_threshold = 0.0002
        self.opt.min_opacity = 0.005
        self.opt.iterations = 30_000
        self.opt.densification_interval = 100
        self.opt.opacity_reset_interval = 3000
        self.opt.densify_from_iter = 500
        self.opt.densify_until_iter = 15_000

        self.opt.deformation_lr = 0.000016  # 0.00016
        self.opt.grid_lr = 0.00016  # 0.0016
        ###
        self.opt.deformation_lr_init = 0.0016  # 0.00016
        self.opt.deformation_lr_final = 0.0000016   # 0.0000016
        self.opt.deformation_lr_delay_mult = 0.01
        self.opt.grid_lr_init = 0.0016  # 0.0016
        self.opt.grid_lr_final = 0.000016   # 0.000016
        self.opt.position_lr_max_steps = 20_000  # 20_000

        self.opt.depth_weight = 1.0
        self.opt.rigid_weight = None
        self.opt.tv_weight = None
        self.opt.smo_weight = None
        self.opt.sep_weight = None
        self.opt.rigid_weight = 1
        self.opt.tv_weight = 0.0002
        self.opt.smo_weight = 0.0002
        self.opt.sep_weight = 0.0001

        self.opt.num_knn = 16
        self.opt.lambda_omega = 200

    def __init__(self, gaussians):
        self.canonical_gaussians = copy.deepcopy(gaussians)
        super().__init__(gaussians)
        self.net = DeformNet()
        self.dataset_st = GroupParams()
        self.dataset_ed = GroupParams()
        self.mlp_scheduler_args = None
        self.grid_scheduler_args = None
        self.mask = None
        self.setup_args_extra()

    @override
    def set_dataset(self, source_path: str, model_path: str, evaluate=True):
        self.dataset_st.sh_degree = 0
        self.dataset_ed.sh_degree = 0
        self.dataset_st.images = "images"
        self.dataset_ed.images = "images"
        self.dataset_st.resolution = -1
        self.dataset_ed.resolution = -1
        self.dataset_st.white_background = False
        self.dataset_ed.white_background = False
        self.dataset_st.data_device = "cuda"
        self.dataset_ed.data_device = "cuda"
        self.dataset_st.eval = False
        self.dataset_ed.eval = False
        self.dataset_st.source_path = os.path.join(os.path.realpath(source_path), 'start')
        self.dataset_ed.source_path = os.path.join(os.path.realpath(source_path), 'end')
        self.dataset_st.model_path = model_path
        self.dataset_ed.model_path = model_path

    @override
    def deform(self):
        canonical_xyz = self.canonical_gaussians.get_xyz
        canonical_rotation = self.canonical_gaussians.get_rotation

        t, q = self.net(canonical_xyz)

        self.gaussians.get_xyz = torch.zeros_like(canonical_xyz)
        self.gaussians.get_rotation_raw = torch.zeros_like(canonical_rotation)
        self.gaussians.get_xyz[:] = canonical_xyz + t
        self.gaussians.get_rotation_raw[:] = quat_mult(q, canonical_rotation)

        self.gaussians.get_scaling_raw = self.canonical_gaussians.get_scaling_raw
        self.gaussians.get_features_dc = self.canonical_gaussians.get_features_dc
        self.gaussians.get_features_rest = self.canonical_gaussians.get_features_rest
        self.gaussians.get_opacity_raw = self.canonical_gaussians.get_opacity_raw
        return t, q

    @override
    def training_setup(self, training_args):
        spatial_lr_scale = self.canonical_gaussians.spatial_lr_scale
        l = [
            {'params': list(self.net.get_grid_parameters()),
             'lr': training_args.grid_lr_init * spatial_lr_scale, "name": "grid"},
            {'params': list(self.net.get_mlp_parameters()),
             'lr': training_args.deformation_lr_init * spatial_lr_scale, "name": "mlp"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init * spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final * spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init * spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final * spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def params_setup(self):
        xyz_st = np.asarray(fetchPly(os.path.join(self.dataset_st.source_path, 'points3d.ply')).points)
        xyz_ed = np.asarray(fetchPly(os.path.join(self.dataset_ed.source_path, 'points3d.ply')).points)
        xyz = np.concatenate((xyz_st, xyz_ed), axis=0) * 1.05
        self.net.set_aabb(xyz.max(axis=0).tolist(), xyz.min(axis=0).tolist())
        self.net.to('cuda')

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mlp":
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "grid":
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr

    def _show_losses(self, iteration: int, losses: dict):
        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 5000, 7000, 15000, 30000]:
            return
        if iteration in [7000, 15000, 30000]:
            self.canonical_gaussians.save_ply(
                os.path.join(self.dataset_ed.model_path, f'point_cloud/iteration_{iteration - 1}/point_cloud.ply'),
                prune=False
            )
            self.gaussians.save_ply(
                os.path.join(self.dataset_ed.model_path, f'point_cloud/iteration_{iteration - 2}/point_cloud.ply'),
                prune=False
            )
            pth_dir = os.path.join(self.dataset_ed.model_path, 'dnet')
            os.makedirs(pth_dir, exist_ok=True)
            torch.save(self.net, os.path.join(pth_dir, f'iteration_{iteration}.pth'))

        loss_msg = f"\nIteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{6}f}"
        print(loss_msg, '\n')

    def _eval_losses(self, render_pkg_st, viewpoint_cam_st, render_pkg_ed, viewpoint_cam_ed):
        losses = {
            'app_st': eval_img_loss(render_pkg_st['render'], viewpoint_cam_st.original_image.cuda(), self.opt),
            'app_ed': eval_img_loss(render_pkg_ed['render'], viewpoint_cam_ed.original_image.cuda(), self.opt),
            'depth_st': eval_depth_loss(render_pkg_st['depth'], viewpoint_cam_st.image_depth.cuda()),
            'depth_ed': eval_depth_loss(render_pkg_ed['depth'], viewpoint_cam_ed.image_depth.cuda()),
            'rig': None,
            'tv': None,
            'smo': None,
            'sep': None,
        }
        weight_st = losses['app_st'].detach() / (losses['app_st'].detach() + losses['app_ed'].detach())
        loss = weight_st * losses['app_st'] + (1 - weight_st) * losses['app_ed']
        weight_st = losses['depth_st'].detach() / (losses['depth_st'].detach() + losses['depth_ed'].detach())
        loss += self.opt.depth_weight * (
            weight_st * losses['depth_st'] + (1 - weight_st) * losses['depth_ed']
        )
        if self.opt.rigid_weight is not None:
            # neighbor_dist, neighbor_indices = knn(self.canonical_gaussians.get_xyz.detach().cpu().numpy(), self.opt.num_knn)
            # neighbor_weight = 2 * np.exp(-self.opt.lambda_omega * neighbor_dist)
            # neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
            # neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
            #
            # curr_rot = self.gaussians.get_rotation
            # prev_rotation_inv = self.canonical_gaussians.get_rotation.detach()
            # prev_rotation_inv[:, 1:] *= -1
            # relative_rot = quat_mult(curr_rot, prev_rotation_inv)
            # rotation = build_rotation(relative_rot)
            #
            # prev_offset = self.canonical_gaussians.get_xyz[neighbor_indices] - self.canonical_gaussians.get_xyz[:, None]
            # curr_offset = self.gaussians.get_xyz[neighbor_indices] - self.gaussians.get_xyz[:, None]
            # curr_offset_in_prev_coord = (rotation.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
            # losses['rig'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offset, neighbor_weight)
            # loss += self.opt.depth_weight * losses['rig']
            pass
        if self.opt.tv_weight is not None:
            losses['tv'] = self.net.eval_tv()
            loss += self.opt.tv_weight * losses['tv']
        if self.opt.sep_weight is not None:
            losses['sep'] = self.net.eval_sep()
            loss += self.opt.sep_weight * losses['sep']
        if self.opt.smo_weight is not None:
            losses['smo'] = self.net.eval_smo()
            loss += self.opt.smo_weight * losses['smo']
        return loss, losses

    def train(self):
        iterations = self.opt.iterations
        bws_st = BWScenes(self.dataset_st, self.canonical_gaussians, is_new_gaussians=True)
        bws_ed = BWScenes(self.dataset_ed, self.canonical_gaussians, is_new_gaussians=False)
        self.canonical_gaussians.training_setup(get_default_args()[2])
        self.training_setup(self.opt)
        self.params_setup()

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(iterations), desc="Training progress")
        for i in range(1, iterations + 1):
            self.update_learning_rate(i)
            # Pick a random Camera from st and ed respectively
            viewpoint_cam_st, background_st = bws_st.pop_black() if (i % 2 == 0) else bws_st.pop_white()
            viewpoint_cam_ed, background_ed = bws_ed.pop_black() if (i % 2 == 0) else bws_ed.pop_white()

            self.deform()

            render_pkg_st = render(viewpoint_cam_st, self.canonical_gaussians, self.pipe, background_st)
            render_pkg_ed = render(viewpoint_cam_ed, self.gaussians, self.pipe, background_ed)
            image, viewspace_point_tensor, visibility_filter, radii \
                = render_pkg_ed["render"], render_pkg_ed["viewspace_points"], render_pkg_ed["visibility_filter"], render_pkg_ed["radii"]

            loss, losses = self._eval_losses(render_pkg_st, viewpoint_cam_st, render_pkg_ed, viewpoint_cam_ed)
            loss.backward()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                # Densification
                if i < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.canonical_gaussians.max_radii2D[visibility_filter] = torch.max(
                        radii[visibility_filter], self.canonical_gaussians.max_radii2D[visibility_filter]
                    )
                    self.canonical_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    # copy or split
                    if i > self.opt.densify_from_iter and i % self.opt.densification_interval == 0:
                        size_threshold = 20 if i > self.opt.opacity_reset_interval else None
                        self.mask = self.canonical_gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, self.opt.min_opacity, bws_st.get_cameras_extent(), size_threshold,
                            auxiliary_attr=self.mask
                        )
                    # opacity reset
                    if i % self.opt.opacity_reset_interval == 0 or (
                            self.dataset_st.white_background and i == self.opt.densify_from_iter):
                        self.canonical_gaussians.reset_opacity()

                if i < iterations:
                    self.optimizer.step()
                    self.canonical_gaussians.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.canonical_gaussians.optimizer.zero_grad(set_to_none=False)

            self._show_losses(i, losses)
        progress_bar.close()

class DMGauFRe(DMCanonical):
    def __init__(self, gaussians):
        super().__init__(gaussians)
        self.mask = None    # movable
        self.opt.cd_thr = -4.5

    @override
    def deform(self):
        canonical_xyz = self.canonical_gaussians.get_xyz[self.mask]
        canonical_rotation = self.canonical_gaussians.get_rotation[self.mask]
        t, q = self.net(canonical_xyz)

        self.gaussians.get_xyz = torch.zeros_like(self.canonical_gaussians.get_xyz)
        self.gaussians.get_rotation_raw = torch.zeros_like(self.canonical_gaussians.get_rotation)
        self.gaussians.get_xyz[:] = self.canonical_gaussians.get_xyz
        self.gaussians.get_rotation_raw[:] = self.canonical_gaussians.get_rotation
        self.gaussians.get_xyz[self.mask] = canonical_xyz + t
        self.gaussians.get_rotation_raw[self.mask] = quat_mult(q, canonical_rotation)

        self.gaussians.get_scaling_raw = self.canonical_gaussians.get_scaling_raw
        self.gaussians.get_features_dc = self.canonical_gaussians.get_features_dc
        self.gaussians.get_features_rest = self.canonical_gaussians.get_features_rest
        self.gaussians.get_opacity_raw = self.canonical_gaussians.get_opacity_raw
        return t, q

    @override
    def params_setup(self):
        xyz_st = np.asarray(fetchPly(os.path.join(self.dataset_st.source_path, 'points3d.ply')).points)
        xyz_ed = np.asarray(fetchPly(os.path.join(self.dataset_ed.source_path, 'points3d.ply')).points)

        x = torch.tensor(xyz_st).unsqueeze(0)
        y = torch.tensor(xyz_ed).unsqueeze(0)
        cd = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, single_directional=True)[0][0]
        cd /= torch.max(cd)
        self.mask = (inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6)) > self.opt.cd_thr)

        x = x[0][self.mask].detach().cpu().numpy()
        storePly(os.path.join(self.dataset_st.model_path, 'pcd-init-m.ply'), x, np.zeros_like(x))

        xyz = np.concatenate((xyz_st, xyz_ed), axis=0) * 1.05
        self.net.set_aabb(xyz.max(axis=0).tolist(), xyz.min(axis=0).tolist())
        self.net.to('cuda')
        self.mask = self.mask.to(device='cuda')

    @override
    def _show_losses(self, iteration: int, losses: dict):
        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 5000, 7000, 15000, 30000]:
            return
        if iteration in [7000, 15000, 30000]:
            self.canonical_gaussians.save_ply(
                os.path.join(self.dataset_ed.model_path, f'point_cloud/iteration_{iteration - 1}/point_cloud.ply'), )
            self.gaussians.save_ply(
                os.path.join(self.dataset_ed.model_path, f'point_cloud/iteration_{iteration - 2}/point_cloud.ply'), )

            pth_dir = os.path.join(self.dataset_ed.model_path, 'dnet')
            os.makedirs(pth_dir, exist_ok=True)
            torch.save(self.net, os.path.join(pth_dir, f'iteration_{iteration}.pth'))
            torch.save(self.mask, os.path.join(pth_dir, f'mask_{iteration}.pt'))

        loss_msg = f"\nIteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{6}f}"
        print(loss_msg, '\n')

class DeformationModel(DeformationBasic):
    def setup_args_extra(self):
        self.opt.deformation_lr_init = 0.000016  # 0.00016
        self.opt.deformation_lr_final = 0.000016   # 0.0000016
        self.opt.deformation_lr_delay_mult = 0.01
        self.opt.grid_lr_init = 0.00016  # 0.0016
        self.opt.grid_lr_final = 0.00016   # 0.000016
        self.opt.position_lr_max_steps = 9_000  # 20_000

        self.opt.iterations = 15_000

        # self.opt.depth_weight = 1.0
        self.opt.depth_weight = 0.1
        self.opt.rigid_weight = None
        self.opt.iso_weight = None
        self.opt.rot_weight = None
        self.opt.tv_weight = None
        self.opt.sep_weight = None
        self.opt.rt_weight = None
        self.opt.rq_weight = None
        self.opt.rigid_weight = 1
        # self.opt.iso_weight = 1
        # self.opt.rot_weight = 10
        self.opt.tv_weight = 0.001     # 0.0002
        self.opt.smo_weight = 0.002     # 0.0002
        self.opt.sep_weight = 0.001    # 0.0001
        # self.opt.rt_weight = 0.01
        # self.opt.rq_weight = 0.01

    def __init__(self, gaussians: GaussianModel):
        super().__init__(gaussians)
        self.net = DeformNet()
        self.original_xyz = self.gaussians.get_xyz.clone().detach()
        self.original_rotation = self.gaussians.get_rotation.clone().detach()
        self.mlp_scheduler_args = None
        self.grid_scheduler_args = None

        self.gaussians.initialize_neighbors(num_knn=16, lambda_omega=1000)
        self.net.set_aabb(self.original_xyz.max(dim=0)[0].tolist(), self.original_xyz.min(dim=0)[0].tolist())
        self.net.to('cuda')
        self.setup_args_extra()

    @override
    def deform(self):
        t, q = self.net(self.original_xyz)
        self.gaussians.get_xyz[:] = self.original_xyz + t
        # self.gaussians.get_rotation_raw[:] = self.original_rotation + q
        # self.gaussians.get_xyz[:] = torch.einsum('nij,nj->ni', build_rotation(q), self.original_xyz) + t
        self.gaussians.get_rotation_raw[:] = quat_mult(q, self.original_rotation)
        return t, q

    @override
    def training_setup(self, training_args):
        spatial_lr_scale = self.gaussians.spatial_lr_scale
        l = [
            {'params': list(self.net.get_mlp_parameters()),
             'lr': training_args.deformation_lr_init * spatial_lr_scale, "name": "mlp"},
            {'params': list(self.net.get_grid_parameters()),
             'lr': training_args.grid_lr_init * spatial_lr_scale, "name": "grid"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init * spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final * spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init * spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final * spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mlp":
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "grid":
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr

    def _show_losses(self, iteration: int, losses: dict):
        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 5000, 7000, 9000, 15000]:
            return
        if iteration in [1, 1000, 5000, 7000, 9000, 15000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply'),
                prune=False
            )
            pth_dir = os.path.join(self.dataset.model_path, 'dnet')
            os.makedirs(pth_dir, exist_ok=True)
            torch.save(self.net, os.path.join(pth_dir, f'iteration_{iteration}.pth'))

        loss_msg = f"\nIteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{6}f}"
        print(loss_msg)
        print()

    def _eval_losses(self, render_pkg, viewpoint_cam, gaussians, t, q):
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'im': eval_img_loss(render_pkg['render'], gt_image, self.opt),
            'd': None,
            'rig': None,
            'iso': None,
            'rot': None,
            'tv': None,
            'sep': None,
            'rt': None,
            'rq': None,
        }
        loss = losses['im']
        if (self.opt.depth_weight is not None) and (viewpoint_cam.image_depth is not None):
            gt_depth = viewpoint_cam.image_depth.cuda()
            losses['d'] = eval_depth_loss(render_pkg['depth'], gt_depth)
            loss += self.opt.depth_weight * losses['d']
        if self.opt.rigid_weight is not None:
            losses['rig'] = eval_rigid_loss(gaussians)
            loss += self.opt.depth_weight * losses['rig']
        if self.opt.iso_weight is not None:
            # losses['iso'] = weighted_l2_loss_v2(q[gaussians.neighbor_indices], q[:, None], gaussians.neighbor_weight)
            losses['iso'] = eval_iso_loss(gaussians)
            loss += self.opt.iso_weight * losses['iso']
        if self.opt.rot_weight is not None:
            losses['rot'] = weighted_l2_loss_v2(t[gaussians.neighbor_indices], t[:, None], gaussians.neighbor_weight)
            loss += self.opt.rot_weight * losses['rot']
        if self.opt.tv_weight is not None:
            losses['tv'] = self.net.eval_tv()
            loss += self.opt.tv_weight * losses['tv']
        if self.opt.sep_weight is not None:
            losses['sep'] = self.net.eval_sep()
            loss += self.opt.sep_weight * losses['sep']
        if self.opt.smo_weight is not None:
            losses['smo'] = self.net.eval_smo()
            loss += self.opt.smo_weight * losses['smo']
        if self.opt.rt_weight is not None:
            losses['rt'] = t.norm(dim=1).mean()
            loss += self.opt.rt_weight * losses['rt']
        if self.opt.rq_weight is not None:
            losses['rq'] = (1 - torch.abs(q[:, 0])).mean()
            loss += self.opt.rq_weight * losses['rq']
        return loss, losses

    @override
    def train(self):
        _ = prepare_output_and_logger(self.dataset)
        iterations = self.opt.iterations
        bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)
        self.training_setup(self.opt)

        progress_bar = tqdm(range(iterations), desc="Training progress")
        ema_loss_for_log = 0.0
        for i in range(1, iterations + 1):
            self.update_learning_rate(i)
            t, q = self.deform()

            # Pick a random Camera
            viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
            loss, losses = self._eval_losses(render_pkg, viewpoint_cam, self.gaussians, t, q)
            loss.backward()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{6}f}"})
                    progress_bar.update(10)
                if i < iterations:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
                    self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach()
            self._show_losses(i, losses)
        progress_bar.close()
