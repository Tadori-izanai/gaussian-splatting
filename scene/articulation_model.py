#
# Created by lxl.
#

import os
import torch
from torch import nn
from tqdm import tqdm
from random import randint

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.general_utils import quat_mult, mat2quat
from utils.loss_utils import eval_losses, show_losses

class ArticulationModel:
    def setup_args(self):
        self.pipe.convert_SHs_python = False
        self.pipe.compute_cov3D_python = False
        self.pipe.debug = False

        self.dataset.sh_degree = 0
        self.dataset.source_path = ""
        self.dataset.model_path = ""
        self.dataset.images = "images"
        self.dataset.resolution = -1
        self.dataset.white_background = False
        self.dataset.data_device = "cuda"
        self.dataset.eval = False

        self.opt.iterations = 10_000
        self.opt.percent_dense = 0.01
        self.opt.lambda_dssim = 0.2
        # self.opt.column_lr = 0.001
        self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.000016
        self.opt.t_lr = 0.00005

        self.opt.rigid_weight = None
        self.opt.rot_weight = None
        self.opt.iso_weight = None
        self.opt.cd_weight = 1
        # self.opt.cd_weight = None
        self.opt.cd_numbers = 10000_0
        self.opt.cd_from_iter = 0
        self.opt.cd_until_iter = self.opt.iterations

    def __init__(self, gaussians: GaussianModel):
        self._column_vec1 = nn.Parameter(
            torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self._column_vec2 = nn.Parameter(
            torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self._t = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self.r_activation = None
        self.gaussians = gaussians
        self.optimizer = None
        self.original_xyz = None
        self.original_rotation = None
        self.dataset = GroupParams()
        self.opt = GroupParams()
        self.pipe = GroupParams()
        self.setup_function()

    def setup_function(self):
        def gram_schmidt(a1: torch.tensor, a2: torch.tensor) -> torch.tensor:
            norm_a1 = torch.norm(a1)
            assert norm_a1 > 0
            b1 = a1 / norm_a1

            b2 = a2 - torch.dot(b1, a2) * b1
            norm_b2 = torch.norm(b2)
            assert norm_b2 > 0
            b2 = b2 / norm_b2

            b3 = torch.linalg.cross(b1, b2)
            return torch.cat([b1.view(3, 1), b2.view(3, 1), b3.view(3, 1)], dim=1)

        self.r_activation = gram_schmidt
        self.gaussians.cancel_grads()
        self.original_xyz = self.gaussians.get_xyz.clone().detach()
        self.original_rotation = self.gaussians.get_rotation.clone().detach()
        self.setup_args()

    @property
    def get_t(self):
        return self._t

    @get_t.setter
    def get_t(self, value):
        with torch.no_grad():
            self._t = value

    @property
    def get_r(self):
        return self.r_activation(self._column_vec1, self._column_vec2)

    @get_r.setter
    def get_r(self, value):
        with torch.no_grad():
            if value.shape == (3, 3):
                self._column_vec1 = value[:, 0]
                self._column_vec2 = value[:, 1]
            elif value.shape == (2, 3):
                self._column_vec1 = value[0]
                self._column_vec2 = value[1]

    def training_setup(self, training_args):
        print(self.gaussians.spatial_lr_scale)
        l = [
            {'params': [self._column_vec1], 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': [self._column_vec2], 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': [self._t], 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def deform(self, mask: torch.tensor):
        r = self.get_r
        r_inv_quat = mat2quat(r.transpose(1, 0))
        self.gaussians.get_xyz[mask] = torch.matmul(self.original_xyz[mask], r) + self.get_t
        self.gaussians.get_rotation_raw[mask] = quat_mult(r_inv_quat, self.original_rotation[mask])
        return self.gaussians

    def _show_losses(self, iteration: int, losses: dict):
        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 4000, 6000, 10000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        print('t:', self.get_t.detach().cpu().numpy())
        print('r:', self.get_r.detach().cpu().numpy())
        if iteration in [20, 200, 2000, 10000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
            )

    def train(self, mask, gt_gaussians=None):
        iterations = self.opt.iterations
        scene = Scene(self.dataset, self.gaussians, is_new_gaussian=False)
        self.training_setup(self.opt)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        viewpoint_stack = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(iterations), desc="Training progress")
        for i in range(1, iterations + 1):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            self.deform(mask)

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()

            loss, losses = eval_losses(self.opt, i, image, gt_image, self.gaussians, gt_gaussians)
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
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r


