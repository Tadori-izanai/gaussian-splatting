#
# Created by lxl.
#

import torch
from torch import nn
import copy

from scene.gaussian_model import GaussianModel
from utils.general_utils import quat_mult, mat2quat

class ArticulationModel:
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
        self.original_xyz = self.gaussians.get_xyz.detach()
        self.original_rotation = self.gaussians.get_rotation.detach()

    def __init__(self, gaussians: GaussianModel):
        self._column_vec1 = nn.Parameter(
            torch.tensor([1, 1, 0], dtype=torch.float, device='cuda').requires_grad_(True)
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
        self.setup_function()

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

    def training_setup(self, training_args):
        l = [
            {'params': [self._column_vec1], 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': [self._column_vec2], 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': [self._t], 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def deform(self, mask):
        r = self.r_activation(self._column_vec1, self._column_vec2)
        r_inv_quat = mat2quat(r.transpose(1, 0))
        self.gaussians.get_xyz[mask] = torch.matmul(self.original_xyz[mask], r) + self._t
        self.gaussians.get_rotation_raw[mask] = quat_mult(r_inv_quat, self.original_rotation[mask])
        return self.gaussians


