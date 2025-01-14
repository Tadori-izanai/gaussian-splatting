import os
import torch
from torch import nn
from tqdm import tqdm
from random import randint

from typing_extensions import override

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene.articulation_model import ArticulationModelBasic
from scene import Scene
from utils.general_utils import quat_mult, mat2quat, mat2quat_batch, inverse_sigmoid
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses
from train import prepare_output_and_logger

class ArticulationModel(ArticulationModelBasic):
    def setup_args_extra(self):
        self.opt.iterations = 15000

        # self.opt.column_lr = 0.005
        # self.opt.t_lr = 0.00005
        self.opt.prob_lr = 0.05
        self.opt.cd_weight = 1.0
        self.opt.bce_weight = None

        # self.opt.mask_thresh = .514
        self.opt.mask_thresh = .85

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
        if iteration in [2000, 4000, 8000, 15000]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration-1}/point_cloud.ply')
            )

        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 4000, 8000, 15000]:
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
        # print('min val in self.get_prob:', torch.min(self.get_prob).detach().cpu().numpy())
        # print('max val in self.get_prob:', torch.max(self.get_prob).detach().cpu().numpy())

    @override
    def _eval_losses(self, image, gt_image, gaussians, gt_gaussians=None):
        losses = {
            'im': eval_img_loss(image, gt_image, self.opt),
            'cd': None,
            'bce': None,
        }
        loss = losses['im']
        if (self.opt.cd_weight is not None) and (gt_gaussians is not None):
            num = gaussians.size() // 2
            definite_gaussians = GaussianModel(0)
            # definite_gaussians.get_xyz = torch.cat(
            #     (
            #         gaussians.get_xyz[:num][self.get_prob > self.opt.mask_thresh],
            #         gaussians.get_xyz[num: 2 * num][self.get_prob < (1 - self.opt.mask_thresh)]
            #     ), dim=0
            # )
            definite_gaussians.get_xyz = gaussians.get_xyz[:num][self.get_prob > self.opt.mask_thresh]

            losses['cd'] = eval_cd_loss(gaussians, gt_gaussians, self.opt.cd_numbers)
            # losses['cd'] = eval_cd_loss(definite_gaussians, gt_gaussians, self.opt.cd_numbers)
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
