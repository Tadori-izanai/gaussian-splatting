#
# Created by lxl.
#

import os
import torch
from torch import nn
from tqdm import tqdm
from random import randint
import copy

from typing_extensions import override

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.general_utils import quat_mult, mat2quat, mat2quat_batch, inverse_sigmoid
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses
from train import prepare_output_and_logger

class ArticulationModelBasic:
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

    def __init__(self, gaussians: GaussianModel, mask: torch.tensor=None):
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
        self.mask = mask
        self.optimizer = None
        self.original_xyz = None
        self.original_rotation = None
        self.dataset = GroupParams()    # ed
        self.opt = GroupParams()
        self.pipe = GroupParams()
        self.setup_function()

    @staticmethod
    def gram_schmidt(a1: torch.tensor, a2: torch.tensor) -> torch.tensor:
        eps = 1e-11
        norm_a1 = torch.norm(a1)
        try:
            assert norm_a1 > eps
        except:
            print(a1, a2)
            exit(0)
        b1 = a1 / norm_a1

        b2 = a2 - torch.dot(b1, a2) * b1
        norm_b2 = torch.norm(b2)
        assert norm_b2 > eps
        b2 = b2 / norm_b2

        b3 = torch.linalg.cross(b1, b2)
        return torch.cat([b1.view(3, 1), b2.view(3, 1), b3.view(3, 1)], dim=1)

    def setup_function(self):
        if self.mask is None:
            self.mask = torch.ones(self.gaussians.size(), dtype=torch.bool)
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

    def set_init_params(self, t, r):
        self._t = nn.Parameter(
            torch.tensor(t, dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self._column_vec1 = nn.Parameter(
            torch.tensor(r[:, 0], dtype=torch.float, device='cuda').requires_grad_(True)
        )
        self._column_vec2 = nn.Parameter(
            torch.tensor(r[:, 1], dtype=torch.float, device='cuda').requires_grad_(True)
        )

    def training_setup(self, training_args):
        print(self.gaussians.spatial_lr_scale)
        l = [
            {'params': [self._column_vec1], 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': [self._column_vec2], 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': [self._t], 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def deform(self):
        """
        Actually, we use the inverse (or transpose) of the rotation matrix
        """
        r = self.get_r
        r_inv_quat = mat2quat(r.transpose(1, 0))
        self.gaussians.get_xyz[self.mask] = torch.matmul(self.original_xyz[self.mask], r) + self.get_t
        self.gaussians.get_rotation_raw[self.mask] = quat_mult(r_inv_quat, self.original_rotation[self.mask])
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

    def train(self, gt_gaussians=None):
        _ = prepare_output_and_logger(self.dataset)
        # if self.mask is None:
        #     self.mask = torch.ones(self.gaussians.size(), dtype=torch.bool)

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

            self.deform()

            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda()

            loss, losses = self._eval_losses(image, gt_image, self.gaussians, gt_gaussians)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss has NaN or inf values")
                exit(0)
            loss.backward()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                # # to prevent zero norm of column_vec1
                # r = self.get_r.detach()
                # self._column_vec1[:] = r[:, 0]
                # self._column_vec2[:] = r[:, 1]
                # #####
                if i < iterations:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
                    self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach()
                    self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach()
            self._show_losses(i, losses)
        progress_bar.close()
        return self.get_t, self.get_r

class ArticulationModelJoint(ArticulationModelBasic):
    def setup_args_extra(self):
        self.dataset_st.sh_degree = 0
        self.dataset_st.source_path = ""
        self.dataset_st.model_path = ""
        self.dataset_st.images = "images"
        self.dataset_st.resolution = -1
        self.dataset_st.white_background = False
        self.dataset_st.data_device = "cuda"
        self.dataset_st.eval = False

        self.dataset_st.source_path = os.path.join(os.path.realpath(self.data_path), 'start')
        self.dataset.source_path = os.path.join(os.path.realpath(self.data_path), 'end')
        self.dataset_st.model_path = self.model_path
        self.dataset.model_path = self.model_path

        self.opt.iterations = 30_000
        self.opt.densification_interval = 100
        self.opt.opacity_reset_interval = 3000
        self.opt.densify_from_iter = 500
        self.opt.densify_until_iter = 15_000
        self.opt.densify_grad_threshold = 0.0002

    def __init__(self, gaussians: GaussianModel, data_path: str, model_path: str, mask: torch.tensor=None):
        self.canonical_gaussians = copy.deepcopy(gaussians)
        super().__init__(gaussians, mask)
        self.data_path = data_path
        self.model_path = model_path
        self.dataset_st = GroupParams()    # self.dataset is ed
        self.setup_args_extra()

    @override
    def deform(self):
        r = self.get_r
        r_inv_quat = mat2quat(r.transpose(1, 0))
        canonical_xyz = self.canonical_gaussians.get_xyz
        canonical_rotation = self.canonical_gaussians.get_rotation

        self.gaussians.get_xyz = torch.zeros_like(canonical_xyz)
        self.gaussians.get_rotation_raw = torch.zeros_like(canonical_rotation)
        self.gaussians.get_xyz[:] = canonical_xyz
        self.gaussians.get_rotation_raw[:] = canonical_rotation
        self.gaussians.get_xyz[self.mask] = torch.matmul(canonical_xyz[self.mask], r) + self.get_t
        self.gaussians.get_rotation_raw[self.mask] = quat_mult(r_inv_quat, canonical_rotation[self.mask])

        self.gaussians.get_scaling_raw = self.canonical_gaussians.get_scaling_raw
        self.gaussians.get_features_dc = self.canonical_gaussians.get_features_dc
        self.gaussians.get_features_rest = self.canonical_gaussians.get_features_rest
        self.gaussians.get_opacity_raw = self.canonical_gaussians.get_opacity_raw
        return self.gaussians

    @override
    def train(self, gt_gaussians=None):
        iterations = self.opt.iterations
        scene_st = Scene(self.dataset_st, self.gaussians, is_new_gaussian=False)
        scene_ed = Scene(self.dataset, self.gaussians, is_new_gaussian=False)
        self.training_setup(self.opt)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        viewpoint_stack_st = None
        viewpoint_stack_ed = None
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(iterations), desc="Training progress")

        for i in range(1, iterations + 1):
            # Pick a random Camera from st and ed respectively
            if not viewpoint_stack_st:
                viewpoint_stack_st = scene_st.getTrainCameras().copy()
            if not viewpoint_stack_ed:
                viewpoint_stack_ed = scene_ed.getTrainCameras().copy()
            viewpoint_cam_st = viewpoint_stack_st.pop(randint(0, len(viewpoint_stack_st) - 1))
            viewpoint_cam_ed = viewpoint_stack_ed.pop(randint(0, len(viewpoint_stack_ed) - 1))

            self.deform()

            losses = {'app_st': None, 'app_ed': None}

            render_pkg = render(viewpoint_cam_st, self.canonical_gaussians, self.pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam_st.original_image.cuda()
            losses['app_st'] = eval_img_loss(image, gt_image, self.opt)

            render_pkg = render(viewpoint_cam_ed, self.gaussians, self.pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam_ed.original_image.cuda()
            losses['app_ed'] = eval_img_loss(image, gt_image, self.opt)

            weight_st = losses['app_st'].detach() / (losses['app_st'].detach() + losses['app_ed'].detach())
            loss = weight_st * losses['app_st'] + (1 - weight_st) * losses['app_ed']
            # loss = losses['app_st'] + losses['app_ed']
            loss.backward()
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                # Densification
                if i < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.canonical_gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.canonical_gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    self.canonical_gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    # copy or split
                    if i > self.opt.densify_from_iter and i % self.opt.densification_interval == 0:
                        size_threshold = 20 if i > self.opt.opacity_reset_interval else None
                        self.mask = self.canonical_gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.005, scene_st.cameras_extent, size_threshold,
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
        return self.get_t, self.get_r

    @override
    def _show_losses(self, iteration: int, losses: dict):
        if iteration in [1000, 2000, 4000, 8000, 15000, 30000]:
            self.canonical_gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
            )
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration + 1}/point_cloud.ply')
            )

        if iteration not in [1, 20, 50, 200, 500, 1000, 2000, 4000, 5999, 8000, 8999, 16000, 30000]:
            return

        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        print('t:', self.get_t.detach().cpu().numpy())
        print('r:', self.get_r.detach().cpu().numpy())
        print('num_gaussians:', self.canonical_gaussians.size())
        print('mean_oacity:', torch.mean(self.canonical_gaussians.get_opacity).detach().cpu().numpy())

