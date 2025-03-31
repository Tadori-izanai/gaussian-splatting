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

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene import BWScenes
from scene.network import DeformNet

from utils.general_utils import get_expon_lr_func, build_rotation, quat_mult, weighted_l2_loss_v2
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

class DeformationModel(DeformationBasic):
    def setup_args_extra(self):
        self.opt.deformation_lr_init = 0.00016
        self.opt.deformation_lr_final = 0.000016   # 0000016
        self.opt.deformation_lr_delay_mult = 0.01
        self.opt.grid_lr_init = 0.0016
        self.opt.grid_lr_final = 0.00016   # 000016
        self.opt.position_lr_max_steps = 90_000

        self.opt.iterations = 15_000

        self.opt.rigid_weight = None
        self.opt.iso_weight = None
        self.opt.rot_weight = None
        self.opt.depth_weight = 1.0
        self.opt.rigid_weight = 1
        self.opt.iso_weight = 1
        # self.opt.rot_weight = 10

    def __init__(self, gaussians: GaussianModel):
        super().__init__(gaussians)
        self.net = DeformNet()
        self.original_xyz = self.gaussians.get_xyz.clone().detach()
        self.original_rotation = self.gaussians.get_rotation.clone().detach()
        self.mlp_scheduler_args = None
        self.grid_scheduler_args = None

        self.gaussians.initialize_neighbors(num_knn=16, lambda_omega=2000)
        self.net.set_aabb(self.original_xyz.max(dim=0)[0].tolist(), self.original_xyz.min(dim=0)[0].tolist())
        self.net.to('cuda')
        self.setup_args_extra()

    @override
    def deform(self):
        t, q = self.net(self.original_xyz)
        self.gaussians.get_xyz[:] = self.original_xyz + t
        self.gaussians.get_rotation_raw[:] = self.original_rotation + q
        # self.gaussians.get_xyz[:] = torch.einsum('nij,nj->ni', build_rotation(q), self.original_xyz) + t
        # self.gaussians.get_rotation_raw[:] = quat_mult(q, self.original_rotation)
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
