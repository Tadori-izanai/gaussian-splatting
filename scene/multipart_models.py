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
from utils.general_utils import quat_mult, mat2quat, inverse_sigmoid, inverse_softmax, \
    strip_symmetric, build_scaling_rotation, eval_quad, decompose_covariance_matrix, build_rotation
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses, eval_cd_loss_sd, \
    eval_knn_opacities_collision_loss, eval_opacity_bce_loss, eval_depth_loss
from train import prepare_output_and_logger

class MPArtModelBasic:
    def setup_args(self):
        self.dataset.sh_degree = 0
        self.dataset.source_path = ""
        self.dataset.model_path = ""
        self.dataset.images = "images"
        self.dataset.resolution = -1
        self.dataset.white_background = False
        self.dataset.data_device = "cuda"
        self.dataset.eval = False

        self.pipe.convert_SHs_python = False
        self.pipe.compute_cov3D_python = False
        self.pipe.debug = False

        self.opt.iterations = 10_000
        self.opt.percent_dense = 0.01
        self.opt.lambda_dssim = 0.2
        self.opt.column_lr = 0.005
        self.opt.t_lr = 0.00005

        self.opt.cd_weight = 1
        self.opt.cd_from_iter = 0
        self.opt.cd_until_iter = self.opt.iterations

    def __init__(self, gaussians: GaussianModel, num_movable: int):
        self.num_movable = num_movable
        self._column_vec1 = [
            nn.Parameter(torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(True))
            for _ in range(self.num_movable)
        ]
        self._column_vec2 = [
            nn.Parameter(torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(True))
            for _ in range(self.num_movable)
        ]
        self._t = [
            nn.Parameter(torch.tensor([0, 0, 0], dtype=torch.float, device='cuda').requires_grad_(True))
            for _ in range(self.num_movable)
        ]
        self.r_activation = None
        self.gaussians = gaussians
        self.optimizer = None
        self.dataset = GroupParams()    # ed
        self.opt = GroupParams()
        self.pipe = GroupParams()
        self.setup_function()

    @staticmethod
    def gram_schmidt(a1: torch.tensor, a2: torch.tensor) -> torch.tensor:
        eps = 1e-11
        norm_a1 = torch.norm(a1)
        b1 = a1 / norm_a1

        b2 = a2 - (b1 @ a2) * b1
        norm_b2 = torch.norm(b2)
        assert norm_b2 > eps
        b2 = b2 / norm_b2

        b3 = torch.linalg.cross(b1, b2)
        return torch.cat([b1.view(3, 1), b2.view(3, 1), b3.view(3, 1)], dim=1)

    def set_dataset(self, source_path: str, model_path: str, evaluate=True):
        self.dataset.eval = evaluate
        self.dataset.source_path = source_path
        self.dataset.model_path = model_path

    def setup_function(self):
        self.r_activation = self.gram_schmidt
        self.gaussians.cancel_grads()
        self.setup_args()

    @property
    def get_t(self):
        return self._t

    @property
    def get_r(self):
        """
        r is actually the transpose of the rotation matrix !!!
        """
        return [self.r_activation(self._column_vec1[k], self._column_vec2[k]) for k in range(self.num_movable)]

    def set_init_params(self, t, r):
        """
        :param t: list of (3, 3) rotation matrices
        :param r: list of (3,) translation vectors
        """
        self._t = [
            nn.Parameter(torch.tensor(tt, dtype=torch.float, device='cuda').requires_grad_(True))
            for tt in t
        ]
        self._column_vec1 = [
            nn.Parameter(torch.tensor(rr[:, 0], dtype=torch.float, device='cuda').requires_grad_(True))
            for rr in r
        ]
        self._column_vec2 = [
            nn.Parameter(torch.tensor(rr[:, 1], dtype=torch.float, device='cuda').requires_grad_(True))
            for rr in r
        ]

    def deform(self):
        pass

    def training_setup(self, training_args):
        l = [
            {'params': self._column_vec1, 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': self._column_vec2, 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': self._t, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def train(self, gt_gaussians=None):
        pass

class GMMArtModel(MPArtModelBasic):
    def setup_args_extra(self):
        self.opt.iterations = 15000
        self.opt.warmup_until_iter = 1000
        self.opt.cd_from_iter = self.opt.warmup_until_iter + 1
        self.opt.cd_until_iter = 7000

        # self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.00005
        self.opt.prob_lr = 0.05
        self.opt.position_lr = 0.00016
        self.opt.scaling_lr = 0.005
        self.opt.opacity_lr = 0.05

        self.opt.cd_weight = 1.0
        self.opt.depth_weight = 1.0

        self.opt.mask_thresh = .85
        # self.opt.trace_r_thresh = 1 + 2 * math.cos(5 / 180 * math.pi)
        self.opt.trace_r_thresh = 1 + 2 * math.cos(10 / 180 * math.pi)

    def __init__(self, gaussians: GaussianModel, num_movable: int):
        super().__init__(gaussians, num_movable)
        self._prob = nn.Parameter(
            torch.zeros(gaussians.size(), dtype=torch.float, device='cuda').requires_grad_(True)
        )
        # GMM parameters below
        self._xyz = nn.Parameter(
            torch.zeros(self.num_movable, 3, dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.zeros(self.num_movable, 3, dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self._rotation_col1 = nn.Parameter(
            torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').repeat(num_movable, 1).requires_grad_(True)
        )
        self._rotation_col2 = nn.Parameter(
            torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').repeat(num_movable, 1).requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.zeros(num_movable, dtype=torch.float, device='cuda').requires_grad_(True)
        )

        def build_inverse_covariance_from_scaling_rotation(scaling, rot_col1, rot_col2):
            ss = torch.diag_embed(1 / (scaling + 1e-8))
            rr = torch.ones(self.num_movable, 3, 3, dtype=rot_col1.dtype, device=rot_col1.device)
            for i in range(self.num_movable):
                rr[i] = self.gram_schmidt(rot_col1[i], rot_col2[i])
            return rr @ ss @ ss @ rr.transpose(1, 2)

        self.prob_activation = torch.sigmoid
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.inverse_covariance_activation = build_inverse_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid

        self.original_xyz = self.gaussians.get_xyz.clone().detach()
        self.original_rotation = self.gaussians.get_rotation.clone().detach()
        self.original_opacity = self.gaussians.get_opacity.clone().detach()
        self.is_revolute = np.array([True for _ in range(self.num_movable)])

        self.gaussians.duplicate(self.num_movable + 1)
        self.setup_args_extra()

    @property
    def get_prob(self):
        return self.prob_activation(self._prob)

    @property
    def get_mu(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_inverse_covariance(self):
        return self.inverse_covariance_activation(self.get_scaling, self._rotation_col1, self._rotation_col2)

    def get_ppp(self):
        quad = eval_quad(self.original_xyz.unsqueeze(1) - self.get_mu, self.get_inverse_covariance)
        ppp = self.get_opacity * torch.exp(-quad)
        return ppp / (ppp.sum(dim=1, keepdim=True) + 1e-10)

    def pred_mp(self):
        return torch.argmax(self.get_ppp(), dim=1)

    def _set_init_probabilities(self, prob=None, mu=None, sigma=None, scaling_modifier=1.0, eps=1e-6):
        if prob is not None:
            prob_raw = inverse_sigmoid(torch.clamp(prob, eps, 1 - eps))
            self._prob = prob_raw.clone().detach().to('cuda').requires_grad_(True)
        if mu is not None:
            self._xyz = mu.clone().detach().to('cuda').requires_grad_(True)
        if sigma is not None:
            scaling, rotation = decompose_covariance_matrix(sigma)
            scaling_raw = self.scaling_inverse_activation(scaling_modifier * scaling)
            scaling_raw = torch.clamp(scaling_raw, -16, 16)
            self._scaling = scaling_raw.clone().detach().to('cuda').requires_grad_(True)
            self._rotation_col1 = rotation[:, :, 0].clone().detach().to('cuda').requires_grad_(True)
            self._rotation_col2 = rotation[:, :, 1].clone().detach().to('cuda').requires_grad_(True)

    def set_init_params(self, model_path: str, scaling_modifier=1.0):
        prob = torch.tensor(np.load(os.path.join(model_path, 'mpp_init.npy')), device='cuda')
        mu = torch.tensor(np.load(os.path.join(model_path, 'mu_init.npy')), device='cuda')
        sigma = torch.tensor(np.load(os.path.join(model_path, 'sigma_init.npy')), device='cuda')
        self._set_init_probabilities(prob, mu, sigma, scaling_modifier)

    @override
    def deform(self):
        num = self.gaussians.size() // (self.num_movable + 1)
        t = self.get_t
        r = self.get_r
        prob = self.get_prob.unsqueeze(-1)
        ppp = self.get_ppp().unsqueeze(-1)

        for k in range(self.num_movable):
            indices = slice(num * (k + 1), num * (k + 2))
            r_inv_quat = mat2quat(r[k].transpose(1, 0))
            self.gaussians.get_xyz[indices] = torch.matmul(self.original_xyz, r[k]) + t[k]
            self.gaussians.get_rotation_raw[indices] = quat_mult(r_inv_quat, self.original_rotation)
            self.gaussians.get_opacity_raw[indices] = inverse_sigmoid(self.original_opacity * prob * ppp[:, k])
        self.gaussians.get_opacity_raw[:num] = inverse_sigmoid((1 - prob) * self.original_opacity)
        return self.gaussians

    @override
    def training_setup(self, training_args):
        l = [
            {'params': self._column_vec1, 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': self._column_vec2, 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': self._t, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': [self._prob], 'lr': training_args.prob_lr, "name": "prob"},
            {'params': [self._xyz], 'lr': training_args.position_lr * self.gaussians.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation_col1], 'lr': training_args.column_lr, "name": "rotation_col1"},
            {'params': [self._rotation_col2], 'lr': training_args.column_lr, "name": "rotation_col2"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self._prob.requires_grad_(False)
        self._xyz.requires_grad_(False)
        self._scaling.requires_grad_(False)
        self._rotation_col1.requires_grad_(False)
        self._rotation_col2.requires_grad_(False)
        self._opacity.requires_grad_(False)

    def _show_losses(self, iteration: int, losses: dict):
        if iteration in [1000, 5000, 9000, 15000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply'),
                prune=False
            )

        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 5000, 7000, 9000, 15000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        for k in np.arange(self.num_movable):
            print(f't{k}:', self.get_t[k].detach().cpu().numpy())
            print(f'r{k}:', self.get_r[k].detach().cpu().numpy())
        print()

    def _eval_losses(self, render_pkg, viewpoint_cam, gaussians, gt_gaussians=None, requires_cd=False):
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'im': eval_img_loss(render_pkg['render'], gt_image, self.opt),
            'bce': None,
            'd': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            num = gaussians.size() // (1 + self.num_movable)
            mp_indices = self.pred_mp()
            definite_gaussians = GaussianModel(0)
            pc_lst = [
                gaussians.get_xyz[num * (k + 1) : num * (k + 2)][
                    (self.get_prob > self.opt.mask_thresh) & (mp_indices == k)
                ] for k in np.arange(self.num_movable)
            ]
            # if self.is_revolute.all():
            #     pc_lst.append(gaussians.get_xyz[:num][self.get_prob < (1 - self.opt.mask_thresh)])
            pc_lst.append(gaussians.get_xyz[:num][self.get_prob < (1 - self.opt.mask_thresh)])
            definite_gaussians.get_xyz = torch.cat(pc_lst, dim=0)
            losses['cd'] = eval_cd_loss_sd(definite_gaussians, gt_gaussians)
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
        bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)
        self.training_setup(self.opt)

        progress_bar = tqdm(range(iterations), desc="Training progress")
        ema_loss_for_log = 0.0
        for i in range(1, iterations + 1):
            requires_cd = self.opt.cd_from_iter <= i <= self.opt.cd_until_iter
            self.deform()

            # Pick a random Camera
            viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
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
                    self._prob[:] = torch.clamp(self._prob, -16, 16)
                    self._opacity[:] = torch.clamp(self._opacity, -16, 16)
                    self._scaling[:] = torch.clamp(self._scaling, -16, 16)
                    self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach()
                    self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
                    self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach()

                if i == self.opt.warmup_until_iter:
                    print('')
                    for k in np.arange(self.num_movable):
                        self.is_revolute[k] = (torch.trace(self.get_r[k]) < self.opt.trace_r_thresh)
                        print(f'Detected part{k} is ' + ('REVOLUTE' if self.is_revolute[k] else 'PRISMATIC'))
                        if self.is_revolute[k]:
                            continue
                        self._column_vec1[k] = nn.Parameter(
                            torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        )
                        self._column_vec2[k] = nn.Parameter(
                            torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        )
                    if self.num_movable > 1:
                        self._xyz.requires_grad_(True)
                        self._scaling.requires_grad_(True)
                        self._rotation_col1.requires_grad_(True)
                        self._rotation_col2.requires_grad_(True)
                        self._opacity.requires_grad_(True)
                    self._prob.requires_grad_(True)
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r

class MPArtModel(MPArtModelBasic):
    def setup_args_extra(self):
        self.opt.iterations = 15000
        self.opt.warmup_until_iter = 1000
        self.opt.cd_from_iter = self.opt.warmup_until_iter + 1
        # self.opt.cd_until_iter = 5000
        self.opt.cd_until_iter = 7000

        # self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.00005
        self.opt.prob_lr = 0.05
        self.opt.cd_weight = 1.0
        # self.opt.depth_weight = None
        self.opt.depth_weight = 1.0
        self.opt.bce_weight = None

        self.opt.mask_thresh = .85
        self.opt.trace_r_thresh = 1 + 2 * math.cos(5 / 180 * math.pi)

    def __init__(self, gaussians: GaussianModel, num_movable: int):
        super().__init__(gaussians, num_movable)
        self._prob = nn.Parameter(
            torch.zeros(gaussians.size(), dtype=torch.float, device='cuda').requires_grad_(True)
        )   # movable part probabilities
        self._ppp = nn.Parameter(
            torch.ones(gaussians.size(), self.num_movable, dtype=torch.float, device='cuda').requires_grad_(True)
        )   # per part probabilities
        self.prob_activation = torch.sigmoid
        self.ppp_activation = lambda x: torch.softmax(x, dim=1)

        self.original_xyz = self.gaussians.get_xyz.clone().detach()
        self.original_rotation = self.gaussians.get_rotation.clone().detach()
        self.original_opacity = self.gaussians.get_opacity.clone().detach()

        # self.gaussians.duplicate(self.num_movable + 1)

        self.is_revolute = np.array([True for _ in range(self.num_movable)])
        self.setup_args_extra()

    @property
    def get_prob(self):
        return self.prob_activation(self._prob)

    @property
    def get_ppp(self):
        return self.ppp_activation(self._ppp)

    def pred_mp(self):
        return torch.argmax(self.get_ppp, dim=1)

    def set_init_probabilities(self, prob=None, ppp=None, eps=1e-6):
        if prob is not None:
            prob_raw = inverse_sigmoid(torch.clamp(prob, eps, 1 - eps))
            self._prob = prob_raw.clone().detach().to('cuda')
            self._prob.requires_grad_(False)
        if ppp is not None:
            ppp_raw = inverse_softmax(torch.clamp(ppp, eps, 1 - eps))
            self._ppp = ppp_raw.clone().detach().to('cuda')
            self._ppp.requires_grad_(False)

    @override
    def deform(self):
        num = self.gaussians.size() // (self.num_movable + 1)
        t = self.get_t
        r = self.get_r
        prob = self.get_prob.unsqueeze(-1)
        ppp = self.get_ppp.unsqueeze(-1)

        for k in range(self.num_movable):
            indices = slice(num * (k + 1), num * (k + 2))
            r_inv_quat = mat2quat(r[k].transpose(1, 0))
            self.gaussians.get_xyz[indices] = torch.matmul(self.original_xyz, r[k]) + t[k]
            self.gaussians.get_rotation_raw[indices] = quat_mult(r_inv_quat, self.original_rotation)
            self.gaussians.get_opacity_raw[indices] = inverse_sigmoid(self.original_opacity * prob * ppp[:, k])
        self.gaussians.get_opacity_raw[:num] = inverse_sigmoid((1 - prob) * self.original_opacity)
        return self.gaussians

    @override
    def training_setup(self, training_args):
        self.gaussians.duplicate(self.num_movable + 1)
        l = [
            {'params': self._column_vec1, 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': self._column_vec2, 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': self._t, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': [self._prob], 'lr': training_args.prob_lr, "name": "prob"},
            {'params': [self._ppp], 'lr': training_args.prob_lr, "name": "ppp"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def _show_losses(self, iteration: int, losses: dict):
        if iteration in [1000, 5000, 9000, 15000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
            )

        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 5000, 7000, 9000, 15000]:
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

    def _eval_losses(self, render_pkg, viewpoint_cam, gaussians, gt_gaussians=None, requires_cd=False):
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'im': eval_img_loss(render_pkg['render'], gt_image, self.opt),
            'd': None,
            'bce': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            num = gaussians.size() // (self.num_movable + 1)
            mp_indices = self.pred_mp()
            definite_gaussians = GaussianModel(0)
            pc_lst = [
                gaussians.get_xyz[num * (k + 1) : num * (k + 2)][
                    (self.get_prob > self.opt.mask_thresh) & (mp_indices == k)
                ] for k in range(self.num_movable)
            ]
            # if self.is_revolute.all():
            #     pc_lst.append(gaussians.get_xyz[:num][self.get_prob < (1 - self.opt.mask_thresh)])
            pc_lst.append(gaussians.get_xyz[:num][self.get_prob < (1 - self.opt.mask_thresh)])
            definite_gaussians.get_xyz = torch.cat(pc_lst, dim=0)
            losses['cd'] = eval_cd_loss_sd(definite_gaussians, gt_gaussians)
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
        bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)
        self.training_setup(self.opt)

        progress_bar = tqdm(range(iterations), desc="Training progress")
        ema_loss_for_log = 0.0
        for i in range(1, iterations + 1):
            requires_cd = (self.opt.cd_from_iter <= i <= self.opt.cd_until_iter)
            self.deform()

            # Pick a random Camera
            viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
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
                    self._prob[:] = torch.clamp(self._prob, -16, 16)
                    self._ppp[:] = torch.clamp(self._ppp, -16, 16)
                    self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach()
                    self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
                    self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach()

                if i == self.opt.warmup_until_iter:
                    print()
                    for k in range(self.num_movable):
                        self.is_revolute[k] = torch.trace(self.get_r[k]) < self.opt.trace_r_thresh
                        print(f'Detected part{k} is ' + ('*REVOLUTE*' if self.is_revolute[k] else '*PRISMATIC*'))
                        if self.is_revolute[k]:
                            continue
                        self._column_vec1[k] = nn.Parameter(
                            torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        )
                        self._column_vec2[k] = nn.Parameter(
                            torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        )
                    self._prob.requires_grad_(True)
                    self._ppp.requires_grad_(True)
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r

class MPArtModelJoint(MPArtModelBasic):
    def setup_args_extra(self):
        self.opt.densify_grad_threshold = 0.0002
        self.opt.min_opacity = 0.005

        self.opt.iterations = 9_000
        # self.opt.iterations = 14_000
        self.opt.densification_interval = 50
        self.opt.opacity_reset_interval = 2000
        self.opt.densify_from_iter = 50
        self.opt.densify_until_iter = 6_000
        # self.opt.densify_until_iter = 11_000

        self.opt.collision_knn = 32
        self.opt.collision_weight = 0.02
        self.opt.collision_from_iter = 1
        # self.opt.collision_from_iter = 5000
        self.opt.collision_until_iter = 10000
        # self.opt.collision_until_iter = self.opt.iterations
        self.opt.collision_after_reset_iter = 500

        self.opt.depth_weight = 1.0

    def __init__(self, gaussians: GaussianModel, num_movable: int):
        self.canonical_gaussians = copy.deepcopy(gaussians)
        super().__init__(gaussians, num_movable)
        self.dataset_st = GroupParams()
        self.dataset_ed = GroupParams()
        self.mask = None
        self.part_indices = None
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

        mask_pre = np.load(os.path.join(model_path, 'mask_pre.npy'))
        part_indices_pre = np.load(os.path.join(model_path, 'part_indices_pre.npy'))
        r_pre = np.load(os.path.join(model_path, 'r_pre.npy'))
        t_pre = np.load(os.path.join(model_path, 't_pre.npy'))
        self.mask = torch.tensor(mask_pre, device='cuda')
        self.part_indices = torch.tensor(part_indices_pre, device='cuda')
        self.set_init_params(t_pre, r_pre)

    @override
    def deform(self):
        r = self.get_r
        t = self.get_t
        canonical_xyz = self.canonical_gaussians.get_xyz
        canonical_rotation = self.canonical_gaussians.get_rotation

        self.gaussians.get_xyz = torch.zeros_like(canonical_xyz)
        self.gaussians.get_rotation_raw = torch.zeros_like(canonical_rotation)
        self.gaussians.get_xyz[:] = canonical_xyz
        self.gaussians.get_rotation_raw[:] = canonical_rotation
        for k in range(self.num_movable):
            r_inv_quat = mat2quat(r[k].transpose(1, 0))
            msk = self.mask & (self.part_indices == k)
            self.gaussians.get_xyz[msk] = torch.matmul(canonical_xyz[msk], r[k]) + t[k]
            self.gaussians.get_rotation_raw[msk] = quat_mult(r_inv_quat, canonical_rotation[msk])

        self.gaussians.get_scaling_raw = self.canonical_gaussians.get_scaling_raw
        self.gaussians.get_features_dc = self.canonical_gaussians.get_features_dc
        self.gaussians.get_features_rest = self.canonical_gaussians.get_features_rest
        self.gaussians.get_opacity_raw = self.canonical_gaussians.get_opacity_raw
        return self.gaussians

    def _show_losses(self, iteration: int, losses: dict):
        if iteration in [1000, 5000, 14000]:
            self.canonical_gaussians.save_ply(
                os.path.join(self.dataset_ed.model_path, f'point_cloud/iteration_{iteration - 1}/point_cloud.ply'),
                prune=False
            )
            self.gaussians.save_ply(
                os.path.join(self.dataset_ed.model_path, f'point_cloud/iteration_{iteration - 2}/point_cloud.ply'),
                prune=False
            )

        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 3000,
                             5000,
                             7000, 10000,
                             14000]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)

        if iteration <= self.opt.collision_from_iter:
            for k in range(self.num_movable):
                print(f't{k}:', self.get_t[k].detach().cpu().numpy())
                print(f'r{k}:', self.get_r[k].detach().cpu().numpy())
        print()

    @override
    def train(self, gt_gaussians=None):
        iterations = self.opt.iterations
        bws_st = BWScenes(self.dataset_st, self.gaussians, is_new_gaussians=False)
        bws_ed = BWScenes(self.dataset_ed, self.gaussians, is_new_gaussians=False)
        self.training_setup(self.opt)

        for k in range(self.num_movable):
            if torch.trace(self.get_r[k]) < 3 * (1 - 1e-6): # revolute
                continue
            self._column_vec1[k].requires_grad_(False)
            self._column_vec2[k].requires_grad_(False)

        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(iterations), desc="Training progress")
        prev_opacity_reset_iter = -114514
        for i in range(1, iterations + 1):
            if i == self.opt.collision_from_iter:
                for k in range(self.num_movable):
                    self._t[k].requires_grad_(False)
                    self._column_vec1[k].requires_grad_(False)
                    self._column_vec2[k].requires_grad_(False)

            # Pick a random Camera from st and ed respectively
            viewpoint_cam_st, background_st = bws_st.pop_black() if (i % 2 == 0) else bws_st.pop_white()
            viewpoint_cam_ed, background_ed = bws_ed.pop_black() if (i % 2 == 0) else bws_ed.pop_white()

            self.deform()

            losses = {'app_st': None, 'app_ed': None, 'depth_st': None, 'depth_ed': None, 'collision': None}
            requires_collision = (i - prev_opacity_reset_iter >= self.opt.collision_after_reset_iter)
            requires_collision &= (self.opt.collision_from_iter <= i <= self.opt.collision_until_iter)

            gt_image = viewpoint_cam_st.original_image.cuda()
            render_pkg = render(viewpoint_cam_st, self.canonical_gaussians, self.pipe, background_st)
            image, viewspace_point_tensor, visibility_filter, radii \
                = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            losses['app_st'] = eval_img_loss(image, gt_image, self.opt)

            if (self.opt.depth_weight is not None) and (viewpoint_cam_st.image_depth is not None):
                gt_depth = viewpoint_cam_st.image_depth.cuda()
                depth = render_pkg['depth']
                losses['depth_st'] = eval_depth_loss(depth, gt_depth)

            gt_image = viewpoint_cam_ed.original_image.cuda()
            render_pkg = render(viewpoint_cam_ed, self.gaussians, self.pipe, background_ed)
            image, viewspace_point_tensor, visibility_filter, radii \
                = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            losses['app_ed'] = eval_img_loss(image, gt_image, self.opt)

            if (self.opt.depth_weight is not None) and (viewpoint_cam_ed.image_depth is not None):
                gt_depth = viewpoint_cam_ed.image_depth.cuda()
                depth = render_pkg['depth']
                losses['depth_ed'] = eval_depth_loss(depth, gt_depth)

            weight_st = losses['app_st'].detach() / (losses['app_st'].detach() + losses['app_ed'].detach())
            loss = weight_st * losses['app_st'] + (1 - weight_st) * losses['app_ed']

            if (self.opt.collision_weight is not None) and requires_collision:
                losses['collision'] = 0
                for k in range(self.num_movable):
                    msk = self.mask & (self.part_indices == k)
                    losses['collision'] = eval_knn_opacities_collision_loss(self.gaussians, msk, k=self.opt.collision_knn)
                    # losses['collision'] += eval_knn_opacities_collision_loss(self.canonical_gaussians, msk, k=self.opt.collision_knn)
                loss += self.opt.collision_weight * losses['collision'] / 1

            if (losses['depth_st'] is not None) and (losses['depth_ed'] is not None):
                weight_st = losses['depth_st'].detach() / (losses['depth_st'].detach() + losses['depth_ed'].detach())
                loss += self.opt.depth_weight * (
                    weight_st * losses['depth_st'] + (1 - weight_st) * losses['depth_ed']
                )

            loss.backward()
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                # if i == self.opt.collision_from_iter:
                #     for k in range(self.num_movable):
                #         self._t[k].requires_grad_(False)
                #         self._column_vec1[k].requires_grad_(False)
                #         self._column_vec2[k].requires_grad_(False)

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
                        self.mask, self.part_indices = self.canonical_gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, self.opt.min_opacity, bws_st.get_cameras_extent(), size_threshold,
                            auxiliary_attr=(self.mask, self.part_indices)
                        )
                    # opacity reset
                    if i % self.opt.opacity_reset_interval == 0 or (
                            self.dataset_st.white_background and i == self.opt.densify_from_iter):
                        self.canonical_gaussians.reset_opacity()
                        prev_opacity_reset_iter = i

                if i < iterations:
                    self.optimizer.step()
                    self.canonical_gaussians.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.canonical_gaussians.optimizer.zero_grad(set_to_none=False)
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r
