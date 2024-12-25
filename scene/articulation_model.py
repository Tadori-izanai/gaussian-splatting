#
# Created by lxl.
#

import os
import torch
from torch import nn
from tqdm import tqdm
from random import randint

from typing_extensions import override

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.general_utils import quat_mult, mat2quat, mat2quat_batch, inverse_sigmoid
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses
from train import prepare_output_and_logger

class ArticulationModelLite:
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
        self.opt.column_lr = 0.005
        self.opt.t_lr = 0.00005

        self.opt.rigid_weight = None
        self.opt.rot_weight = None
        self.opt.iso_weight = None
        self.opt.cd_weight = 1
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

    @staticmethod
    def gram_schmidt(a1: torch.tensor, a2: torch.tensor) -> torch.tensor:
        eps = 1e-11
        norm_a1 = torch.norm(a1)
        assert norm_a1 > eps
        b1 = a1 / norm_a1

        b2 = a2 - torch.dot(b1, a2) * b1
        norm_b2 = torch.norm(b2)
        assert norm_b2 > eps
        b2 = b2 / norm_b2

        b3 = torch.linalg.cross(b1, b2)
        return torch.cat([b1.view(3, 1), b2.view(3, 1), b3.view(3, 1)], dim=1)

    def setup_function(self):
        self.r_activation = self.gram_schmidt
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

    def deform(self, mask: torch.tensor=None):
        if mask is None:
            mask = torch.ones(self.gaussians.size(), dtype=torch.bool)
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

    def _eval_losses(self, image, gt_image, gaussians, gt_gaussians=None):
        losses = {
            'im': eval_img_loss(image, gt_image, self.opt),
            'cd': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None):
            losses['cd'] = eval_cd_loss(gaussians, gt_gaussians, self.opt.cd_numbers)
            loss += self.opt.cd_weight * losses['cd']
        return loss, losses

    def train(self, mask=None, gt_gaussians=None):
        _ = prepare_output_and_logger(self.dataset)
        if mask is None:
            mask = torch.ones(self.gaussians.size(), dtype=torch.bool)

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

            loss, losses = self._eval_losses(image, gt_image, self.gaussians, gt_gaussians)
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
                    # to prevent zero norm of column_vec1
                    # r = self.get_r
                    # self._column_vec1[:] = r[:, 0]
                    # self._column_vec2[:] = r[:, 1]
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r

class ArticulationModel(ArticulationModelLite):
    def setup_args_extra(self):
        self.opt.iterations = 15000

        # self.opt.column_lr = 0.005
        self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.00005
        # self.opt.t_lr = 0.00002

        self.opt.prob_lr = 0.05

        # self.opt.cd_weight = 0.5
        self.opt.cd_weight = 1.0
        self.opt.bce_weight = None
        # self.opt.bce_weight = 0.001

    def __init__(self, gaussians: GaussianModel):
        super().__init__(gaussians)
        self._prob = nn.Parameter(
            torch.zeros(gaussians.size(), dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self.prob_activation = torch.sigmoid
        self.original_opacity = self.gaussians.get_opacity.clone().detach()
        self.gaussians.duplicate()
        self.setup_args_extra()

    @property
    def get_prob(self):
        return self.prob_activation(self._prob)

    @override
    def deform(self, mask: torch.tensor=None):
        num = self.gaussians.size() // 2
        r = self.get_r
        r_inv_quat = mat2quat(r.transpose(1, 0))
        prob = self.get_prob.unsqueeze(-1)

        self.gaussians.get_xyz[:num] = torch.matmul(self.original_xyz, r) + self.get_t
        self.gaussians.get_rotation_raw[:num] = quat_mult(r_inv_quat, self.original_rotation)
        self.gaussians.get_opacity_raw[:num] = inverse_sigmoid(prob * self.original_opacity)
        self.gaussians.get_opacity_raw[num: 2 * num] = inverse_sigmoid((1 - prob) * self.original_opacity)
        return self.gaussians

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
        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 4000, 8000, 15000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        print('t:', self.get_t.detach().cpu().numpy())
        print('r:', self.get_r.detach().cpu().numpy())
        print('_column_vec1:', self._column_vec1.detach().cpu().numpy())
        print('_column_vec2:', self._column_vec2.detach().cpu().numpy())
        print('min val in self.get_prob:', torch.min(self.get_prob).detach().cpu().numpy())
        print('max val in self.get_prob:', torch.max(self.get_prob).detach().cpu().numpy())
        if iteration in [2000, 4000, 8000, 15000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
            )

    @override
    def _eval_losses(self, image, gt_image, gaussians, gt_gaussians=None):
        losses = {
            'im': eval_img_loss(image, gt_image, self.opt),
            'cd': None,
            'bce': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None):
            losses['cd'] = eval_cd_loss(gaussians, gt_gaussians, self.opt.cd_numbers)
            loss += self.opt.cd_weight * losses['cd']
        if self.opt.bce_weight is not None:
            losses['bce'] = self._eval_bce_loss()
            loss += self.opt.bce_weight * losses['bce']
        return loss, losses

    def _eval_bce_loss(self):
        eps = 1e-5      # cannot be smaller
        p = self.get_prob
        p = torch.clamp(p, min=eps, max=1 - eps)
        return torch.nn.functional.binary_cross_entropy(p, p, reduction='mean')

class ArticulationModel0(ArticulationModelLite):
    def setup_args_extra(self):
        self.opt.iterations = 30_000
        # self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.00005
        self.opt.column_lr = 0.005
        self.opt.t_lr = 0.00005
        self.opt.rel_column_lr = 0.005
        self.opt.rel_t_lr = 0.00005
        self.opt.lam_t = 100
        self.opt.reg_weight = 1
        # self.opt.cd_weight = 1
        self.opt.cd_weight = 0.2
        self.opt.rigid_r_weight = 100
        self.opt.rigid_t_weight = 100

    @staticmethod
    def gram_schmidt(a1: torch.tensor, a2: torch.tensor) -> torch.tensor:
        if a1.dim() == 1:
            a1 = a1.unsqueeze(0)
            a2 = a2.unsqueeze(0)

        norm_a1 = torch.norm(a1, dim=1, keepdim=True)
        assert torch.all(norm_a1 > 0), "a1 contains zero vectors"
        b1 = a1 / norm_a1

        dot_product = torch.sum(b1 * a2, dim=1, keepdim=True)
        b2 = a2 - dot_product * b1
        norm_b2 = torch.norm(b2, dim=1, keepdim=True)
        assert torch.all(norm_b2 > 0), "a2 is not linearly independent of a1"
        b2 = b2 / norm_b2

        b3 = torch.cross(b1, b2, dim=1)
        result = torch.cat([b1.unsqueeze(2), b2.unsqueeze(2), b3.unsqueeze(2)], dim=2)
        return result.squeeze(0) if result.shape[0] == 1 else result

    def __init__(self, gaussians: GaussianModel):
        super().__init__(gaussians)
        num = gaussians.size()
        self._rel_column_vec1 = nn.Parameter(
            torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').repeat(num, 1).requires_grad_(True)
        )
        self._rel_column_vec2 = nn.Parameter(
            torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').repeat(num, 1).requires_grad_(True)
        )
        self._rel_t = nn.Parameter(
            torch.zeros((num, 3), dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self.setup_args_extra()
        self.gaussians.initialize_neighbors(num_knn=20, lambda_omega=20, simple=True)

    @property
    def get_rel_t(self):
        return self._rel_t

    @property
    def get_rel_r(self):
        return self.r_activation(self._rel_column_vec1, self._rel_column_vec2)

    @override
    def training_setup(self, training_args):
        print(self.gaussians.spatial_lr_scale)
        l = [
            {'params': [self._column_vec1], 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': [self._column_vec2], 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': [self._t], 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': [self._rel_column_vec1], 'lr': training_args.rel_column_lr, 'name': 'rel_column_vec1'},
            {'params': [self._rel_column_vec2], 'lr': training_args.rel_column_lr, 'name': 'rel_column_vec2'},
            {'params': [self._rel_t], 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "rel_t"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @override
    def deform(self, mask: torch.tensor=None):
        if mask is None:
            mask = torch.ones(self.gaussians.size(), dtype=torch.bool)
        r = self.get_r
        rel_r = self.get_rel_r[mask]
        r_inv_quat = mat2quat(r.transpose(1, 0))
        rel_r_quat = mat2quat_batch(rel_r)

        # self.gaussians.get_xyz[mask] = torch.bmm(
        #     rel_r.transpose(1, 2), (
        #         torch.matmul(self.original_xyz[mask], r) + self.get_t - self.get_rel_t[mask]
        #     ).unsqueeze(-1)
        # ).squeeze(-1)
        self.gaussians.get_xyz[mask] = torch.bmm(
            rel_r.transpose(1, 2), torch.matmul(self.original_xyz[mask], r).unsqueeze(-1)
        ).squeeze(-1) + self.get_t - self.get_rel_t[mask]

        self.gaussians.get_rotation_raw[mask] = quat_mult(
            r_inv_quat, quat_mult(rel_r_quat, self.original_rotation[mask])
        )
        return self.gaussians

    @override
    def _eval_losses(self, image, gt_image, gaussians, gt_gaussians=None):
        losses = {
            'im': eval_img_loss(image, gt_image, self.opt),
            'cd': None,
            'reg_r': None,
            'lam_reg_t': None,
            'rigid_r': None,
            'rigid_t': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None):
            losses['cd'] = eval_cd_loss(gaussians, gt_gaussians, self.opt.cd_numbers)
            loss += self.opt.cd_weight * losses['cd']
        if self.opt.reg_weight is not None:
            losses['reg_r'], losses['lam_reg_t'] = self._eval_reg_loss()
            loss += self.opt.reg_weight * (losses['reg_r'] + losses['lam_reg_t'])
        if (self.opt.rigid_r_weight is not None) and (self.opt.rigid_t_weight is not None):
            losses['rigid_r'], losses['rigid_t'] = self._eval_rigid_loss()
            loss += self.opt.rigid_r_weight * losses['rigid_r']
            loss += self.opt.rigid_t_weight * losses['rigid_t']
        return loss, losses

    def _eval_reg_loss(self):
        num = self.gaussians.size()
        # r = self.get_r
        # t = self.get_t
        r = self.get_r.detach()
        t = self.get_t.detach()
        rel_r = self.get_rel_r
        rel_t = self.get_rel_t
        identity = torch.eye(3, dtype=torch.float, device='cuda')
        reg_r = torch.mean(
            (
                torch.bmm(rel_r.transpose(1, 2), r.unsqueeze(0).expand(num, -1, -1)) - identity
            ).norm(dim=(1, 2)) * (rel_r - identity).norm(dim=(1, 2))
        )
        lam_reg_t = self.opt.lam_t * torch.mean(
            torch.norm(rel_t - t, dim=1) * torch.norm(rel_t, dim=1)
        ) / (self.gaussians.spatial_lr_scale ** 2)
        return reg_r, lam_reg_t

    def _eval_rigid_loss(self):
        rel_t = self.get_rel_t
        rel_r = self.get_rel_r
        identity = torch.eye(3, dtype=torch.float, device='cuda')
        w = self.gaussians.neighbor_weight
        indices = self.gaussians.neighbor_indices

        t_offset = (rel_t[indices] - rel_t[:, None]).norm(dim=-1)
        rigid_t = torch.mean(t_offset * w)
        r_offset = (rel_r[indices] * rel_r.transpose(1, 2)[:, None] - identity).norm(dim=(-1, -2))
        rigid_r = torch.mean(r_offset * w)
        return rigid_r, rigid_t

    @override
    def _show_losses(self, iteration: int, losses: dict):
        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 4000, 8000, 15000, 30000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        print('t:', self.get_t.detach().cpu().numpy())
        print('r:', self.get_r.detach().cpu().numpy())
        if iteration in [2000, 4000, 8000, 15000, 30000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
            )

