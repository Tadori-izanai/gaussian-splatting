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
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

os.environ['USE_KEOPS'] = '1'
from geomloss import SamplesLoss

from typing_extensions import override

from gaussian_renderer import render
from arguments import GroupParams
from scene.gaussian_model import GaussianModel
from scene import BWScenes
from scene.dataset_readers import fetchPly, storePly
from utils.general_utils import quat_mult, mat2quat, inverse_sigmoid, inverse_softmax, \
    strip_symmetric, build_scaling_rotation, eval_quad, decompose_covariance_matrix, build_rotation, \
    find_close, find_files_with_suffix, kl_divergence_gaussian, value_to_rgb
from utils.loss_utils import eval_losses, eval_img_loss, eval_cd_loss, show_losses, eval_cd_loss_sd, \
    eval_knn_opacities_collision_loss, eval_opacity_bce_loss, eval_depth_loss, sample_pts
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH

from utils.dual_quaternion import quaternion_mul, matrix_to_quaternion, dual_quaternion_apply

from train import prepare_output_and_logger

from plyfile import PlyData, PlyElement

COLORS = [
    (1, 0, 0),  # r
    (0, 1, 0),  # g
    (0, 0, 1),  # b
    (1, 1, 0),  # yellow
    (1, 0, 1),  # magenta
    (0, 1, 1),  # cyan
    (1, .5, 0), # orange
]

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
        self._c = [
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
        # return self._t
        return [
            -torch.matmul(c, r) + c + t
            for c, t, r in zip(self._c, self._t, self.get_r)
        ]

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

    def deform(self, iteration: int):
        pass

    def training_setup(self, training_args):
        l = [
            {'params': self._column_vec1, 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': self._column_vec2, 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': self._t, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': self._c, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "c"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def train(self, gt_gaussians=None):
        pass

class GMMArtModel(MPArtModelBasic):
    def setup_args_extra(self):
        # self.opt.iterations = 10000
        self.opt.iterations = 15000
        self.opt.warmup_until_iter = 1000

        self.opt.cd_from_iter = 1
        self.opt.cd_until_iter = 7000
        self.opt.cd_from_weight = 0
        self.opt.cd_until_weight = 0.5

        self.opt.sgd_interval = 2

        # self.opt.column_lr = 0.005
        self.opt.column_lr = 0.001
        # self.opt.t_lr = 0.00005
        self.opt.prob_lr = 0.05
        self.opt.position_lr = 0.00016
        self.opt.scaling_lr = 0.005
        self.opt.opacity_lr = 0.05

        # self.opt.depth_scaling = 100
        self.opt.depth_scaling = 1

        self.opt.depth_weight = None
        self.opt.cd_weight = None
        self.opt.center_weight = None
        self.opt.ppp_weight = None
        self.opt.cr_weight = None
        self.opt.ppp_weight_ed = None
        self.opt.kl_weight = None
        self.opt.sd_weight = None

        self.opt.depth_weight = 1.0
        self.opt.cd_weight = 1.0
        self.opt.center_weight = 0.1
        self.opt.ppp_weight = 0.1
        # self.opt.ppp_weight_ed = 0.1
        # self.opt.kl_weight = 1e-5
        # self.opt.sd_weight = 1.0

        self.opt.mask_thresh = .85
        self.opt.trace_r_thresh = 1 + 2 * math.cos(5 / 180 * math.pi)

    def __init__(self, gaussians: GaussianModel, num_movable: int, new_scheme=True):
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
        self._p = 1.0
        self.ppp = None

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
        self.original_gaussians = copy.deepcopy(self.gaussians)
        self.is_revolute = np.array([True for _ in range(self.num_movable)])
        
        self.pcd_gt = None  # end state pcd from depth map
        self.pcds = []      # start state clustered pcds
        self.pcds_sample = []
        self.pcds_deformed = []
        self.pcd_knn_indices = []

        self.ed_knn_indices = []

        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

        self.new_scheme = new_scheme
        self.gaussians.duplicate(2 if new_scheme else self.num_movable + 1)
        self.setup_args_extra()

    @property
    def get_prob(self):
        return self.prob_activation(self._prob)

    @property
    def get_mu(self):
        return self._xyz
    
    @property
    def get_rotation(self):
        rr = torch.ones(self.num_movable, 3, 3, dtype=self._rotation_col1.dtype, device=self._rotation_col1.device)
        for i in range(self.num_movable):
            rr[i] = self.gram_schmidt(self._rotation_col1[i], self._rotation_col2[i])
        return rr

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_inverse_covariance(self):
        return self.inverse_covariance_activation(self.get_scaling, self._rotation_col1, self._rotation_col2)

    def cosine_anneal(self, step, final_step=-1, start_step=0, start_value=1.0, final_value=0.1):
        if final_step == -1:
            final_step = self.opt.iterations

        if step < start_step:
            value = start_value
        elif step >= final_step:
            value = final_value
        else:
            a = 0.5 * (start_value - final_value)
            b = 0.5 * (start_value + final_value)
            progress = (step - start_step) / (final_step - start_step)
            value = a * math.cos(math.pi * progress) + b
        return value

    def _cal_relative_pos(self, x, mu=None, rot=None, scale=None):
        mu = self.get_mu if mu is None else mu      # x [N, 3], mu [K, 3]
        rot = self.get_rotation if rot is None else rot     # rot [K, 3, 3]
        scale = self.get_scaling if scale is None else scale # scale [K, 3]
        # [N, K, 3]
        return torch.einsum('kji,nkj->nki', rot, x.unsqueeze(1) - mu) / scale

    def get_ppp(self, pts=None, deformed=False, tau=1.0, eps=1e-8):
        if (self.ppp is not None) and (pts is None):
            return self.ppp
        save = (pts is None)
        if deformed:
            assert pts is not None
            r = torch.stack(self.get_r).to(dtype=self.get_mu.dtype)
            t = torch.stack(self.get_t).to(dtype=self.get_mu.dtype)
            mu = torch.einsum('kji,kj->ki', r, self.get_mu) + t
            rot = r.transpose(1, 2) @ self.get_rotation
            rel_pos = self._cal_relative_pos(pts, mu, rot, self.get_scaling)
        else:
            if pts is None:
                pts = self.original_xyz
            rel_pos = self._cal_relative_pos(pts)
        quad = torch.sum(rel_pos ** 2, dim=-1) ** self._p  # [N, K]
        ppp = self.get_opacity * torch.exp(-quad / tau)
        ppp = ppp.clamp(eps, 1 - eps)
        ppp /= ppp.sum(dim=1, keepdim=True)

        self.ppp = ppp if save else self.ppp
        return ppp

    def get_ppp_naive(self, pts=None, deformed=False, tau=1.0, eps=1e-8):
        inv_cov = self.get_inverse_covariance
        mu = self.get_mu
        if deformed:
            assert pts is not None
            r = torch.stack(self.get_r).to(dtype=mu.dtype)
            t = torch.stack(self.get_t).to(dtype=mu.dtype)
            # r = torch.tensor(np.load(os.path.join(self.dataset.model_path, 'r_final.npy'))).to(device='cuda', dtype=mu.dtype)
            # t = torch.tensor(np.load(os.path.join(self.dataset.model_path, 't_final.npy'))).to(device='cuda', dtype=mu.dtype)
            inv_cov = r.transpose(1, 2) @ inv_cov @ r
            mu = torch.einsum('kji,kj->ki', r, mu) + t
        if pts is None:
            pts = self.original_xyz
        quad = eval_quad(pts.unsqueeze(1) - mu, inv_cov) ** self._p
        ppp = self.get_opacity * torch.exp(-quad / tau)
        ppp = ppp.clamp(eps, 1 - eps)
        return ppp / ppp.sum(dim=1, keepdim=True)

    def pred_mp(self):
        return torch.argmax(self.get_ppp(), dim=1)

    @override
    def set_dataset(self, source_path: str, model_path: str, evaluate=True, thr=-5):
        super().set_dataset(source_path, model_path, evaluate)
        xyz_ed = np.asarray(fetchPly(os.path.join(source_path, 'points3d-100k.ply')).points)
        xyz_st = np.asarray(fetchPly(os.path.join(source_path, '../start/points3d.ply')).points)
        y = torch.tensor(xyz_st, device='cuda').unsqueeze(0)
        y = sample_pts(y, 10_100_000)
        x = torch.tensor(xyz_ed, device='cuda').unsqueeze(0)
        cd = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, single_directional=True)[0][0]
        cd /= torch.max(cd)
        mask = inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6)) > thr

        edm = xyz_ed[mask.detach().cpu().numpy()]
        storePly(os.path.join(model_path, f'clustering/points3d-edm.ply'), edm, np.zeros_like(edm))
        self.pcd_gt = torch.tensor(edm, device='cuda')

        cluster_dir = os.path.join(model_path, 'clustering/clusters')
        for i in np.arange(20):
            ply_file = os.path.join(cluster_dir, f'points3d_{i}.ply')
            if os.path.exists(ply_file):
                pcd = torch.tensor(np.asarray(fetchPly(ply_file).points), device='cuda')
                pcd = sample_pts(pcd, len(pcd) * 100_000 // len(xyz_st))
                self.pcds.append(pcd)
                self.pcds_deformed.append(pcd)
                self.pcds_sample.append(sample_pts(pcd, len(pcd) // 10))
                # self.pcds_sample.append(sample_pts(pcd, 1000))
            if len(self.pcds) == self.num_movable:
                break
        assert len(self.pcds) == self.num_movable

        for k in range(self.num_movable):
            p1 = self.pcds_sample[k].unsqueeze(0)
            p2 = self.original_xyz.detach().unsqueeze(0)
            _, indices, _ = knn_points(p1, p2, K=1)
            self.pcd_knn_indices.append(indices.flatten())

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
        self._c = [
            nn.Parameter(torch.tensor(c, dtype=torch.float, device='cuda').requires_grad_(True))
            for c in mu
        ]

    def _get_slot_deform(self):
        qrs = []
        qds = []
        for r, t in zip(self.get_r, self.get_t):
            r = r.transpose(1, 0)
            qr = matrix_to_quaternion(r)
            t0 = torch.cat([torch.zeros(1).to(qr.device), t])
            qd = 0.5 * quaternion_mul(t0, qr)
            qrs.append(qr)
            qds.append(qd)
        qrs, qds = torch.stack(qrs), torch.stack(qds)
        return qrs, qds

    def _dual_quat_deform(self, iteration=-1):
        ppp = self.get_ppp(tau=self.cosine_anneal(iteration))
        qr, qd = self._get_slot_deform()
        qr = torch.einsum('nk,kl->nl', ppp, qr.to(dtype=ppp.dtype))  # [N, 4]
        qd = torch.einsum('nk,kl->nl', ppp, qd.to(dtype=ppp.dtype))  # [N, 4]
        xyz = dual_quaternion_apply((qr, qd), self.original_xyz)
        rot = quaternion_mul(qr, self.original_rotation)
        return xyz, rot

    def joint_pred(self):
        def is_likely_prismatic(diff: np.ndarray, thresh=0.02) -> bool:
            if len(diff) < 10:
                return False
            std = np.linalg.norm(np.std(diff, axis=0, ddof=1))
            mean = np.linalg.norm(diff, axis=1).mean()
            is_prismatic = (std / (mean + 1e-5)) < thresh
            # if is_prismatic:
            #     print(len(diff))
            #     print(std, mean)
            return is_prismatic

        corr_path = os.path.join(self.dataset.source_path, '../correspondence_loftr/no_filter')
        prismatic = []
        for npz_file in find_files_with_suffix(corr_path, '.npz'):
            corr = np.load(os.path.join(corr_path, npz_file), allow_pickle=True)['data'][0]
            xyz_st, xyz_ed = corr['src_world'], corr['tgt_world']
            for k in np.arange(self.num_movable):
                if k in prismatic:
                    continue
                indices = find_close(self.pcds[k].detach().cpu().numpy(), xyz_st, threshold=0.016)
                displacement = xyz_ed[indices] - xyz_st[indices]
                if is_likely_prismatic(displacement):
                    print('predicted joint', k, 'as prismatic!')
                    self._column_vec1[k].requires_grad_(False)
                    self._column_vec2[k].requires_grad_(False)
                    prismatic.append(k)
        print('done with joint pred.\n')
        exit(0)

    @override
    def deform(self, iteration: int):
        t = self.get_t
        r = self.get_r
        prob = self.get_prob.unsqueeze(-1)
        ppp = self.get_ppp().unsqueeze(-1)

        if self.new_scheme:
            num = self.gaussians.size() // 2
            xyz, rot = self._dual_quat_deform()
            # xyz, rot = self._dual_quat_deform(iteration)
            self.gaussians.get_xyz[num:] = xyz
            self.gaussians.get_rotation_raw[num:] = rot
            self.gaussians.get_opacity_raw[num:] = inverse_sigmoid(prob * self.original_opacity)
            self.gaussians.get_opacity_raw[:num] = inverse_sigmoid((1 - prob) * self.original_opacity)
            for k in range(self.num_movable):
                self.pcds_deformed[k] = torch.matmul(self.pcds[k], r[k]) + t[k]
            return self.gaussians

        num = self.gaussians.size() // (self.num_movable + 1)
        for k in range(self.num_movable):
            indices = slice(num * (k + 1), num * (k + 2))
            r_inv_quat = mat2quat(r[k].transpose(1, 0))
            # mask_visible = self.gaussians.get_opacity_raw[indices].squeeze(-1) > 0.005
            mask_visible = slice(None)
            self.gaussians.get_xyz[indices][mask_visible] = torch.matmul(self.original_xyz[mask_visible], r[k]) + t[k]
            self.gaussians.get_rotation_raw[indices][mask_visible] = quat_mult(r_inv_quat, self.original_rotation[mask_visible])
            self.gaussians.get_opacity_raw[indices] = inverse_sigmoid(self.original_opacity * prob * ppp[:, k])
            self.pcds_deformed[k] = torch.matmul(self.pcds[k], r[k]) + t[k]
        self.gaussians.get_opacity_raw[:num] = inverse_sigmoid((1 - prob) * self.original_opacity)
        return self.gaussians

    def save_ppp_vis(self, path: str):
        mkdir_p(os.path.dirname(path))
        ppp = self.get_ppp()
        fused_color = torch.zeros(self.original_gaussians.size(), 3)
        for k in range(self.num_movable):
            c = torch.tensor(COLORS[k % len(COLORS)])
            fused_color += ppp[:, k].unsqueeze(1).cpu() * c

        self.original_gaussians.save_vis(path, fused_color)

    def save_mpp_vis(self, path: str):
        mkdir_p(os.path.dirname(path))
        mpp = self.get_prob
        fused_color = value_to_rgb(mpp)
        self.original_gaussians.save_vis(path, fused_color)

    def save_pp_vis(self, path: str):
        mkdir_p(os.path.dirname(path))
        ppp = self.get_ppp()
        mpp = self.get_prob
        fused_color = torch.zeros(self.original_gaussians.size(), 3)
        for k in range(self.num_movable):
            c = torch.tensor(COLORS[k % len(COLORS)])
            fused_color += ppp[:, k].unsqueeze(1).cpu() * c
        fused_color[mpp < self.opt.mask_thresh] = 0
        self.original_gaussians.save_vis(path, fused_color)

    def save_all_vis(self, iteration=-20):
        pcd_dir = self.dataset.model_path
        self.save_mpp_vis(os.path.join(pcd_dir, f'point_cloud/iteration_{iteration + 2}/point_cloud.ply'))
        self.save_ppp_vis(os.path.join(pcd_dir, f'point_cloud/iteration_{iteration + 1}/point_cloud.ply'))
        self.save_pp_vis(os.path.join(pcd_dir, f'point_cloud/iteration_{iteration}/point_cloud.ply'))

    @override
    def training_setup(self, training_args):
        l = [
            {'params': self._column_vec1, 'lr': training_args.column_lr, "name": "column_vec1"},
            {'params': self._column_vec2, 'lr': training_args.column_lr, "name": "column_vec2"},
            {'params': self._t, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "t"},
            {'params': self._c, 'lr': training_args.t_lr * self.gaussians.spatial_lr_scale, "name": "c"},
            {'params': [self._prob], 'lr': training_args.prob_lr, "name": "prob"},
            {'params': [self._xyz], 'lr': training_args.position_lr * self.gaussians.spatial_lr_scale, "name": "xyz"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation_col1], 'lr': training_args.column_lr, "name": "rotation_col1"},
            {'params': [self._rotation_col2], 'lr': training_args.column_lr, "name": "rotation_col2"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self._prob.requires_grad_(False)
        if self.num_movable == 1:
            self._xyz.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation_col1.requires_grad_(False)
            self._rotation_col2.requires_grad_(False)
            self._opacity.requires_grad_(False)
        for c in self._c:
            c.requires_grad_(False)

        # k = 0
        # for v1, v2 in zip(self._column_vec1, self._column_vec2):
        #     if k == 2:
        #         k += 1
        #         continue
        #     v1.requires_grad_(False)
        #     v2.requires_grad_(False)
        #     k += 1

    def _show_losses(self, iteration: int, losses: dict):
        if iteration == self.opt.warmup_until_iter:
            self.save_ppp_vis(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration-1}/point_cloud.ply')
            )
        if iteration in [1000, 5000, 9000, 15000, self.opt.iterations]:
            self.gaussians.save_ply(
                os.path.join(self.dataset.model_path, f'point_cloud/iteration_{iteration}/point_cloud.ply'),
                prune=False
            )

        if iteration not in [1, 50, 500, 1001, 2000, 5000, 7000, 9000, 12000, 15000, self.opt.iterations]:
            return
        loss_msg = f"\niteration {iteration}:"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
        for k in np.arange(self.num_movable):
            print(f't{k}:', self.get_t[k].detach().cpu().numpy())
            print(f'r{k}:', self.get_r[k].detach().cpu().numpy())
            print(f'_c{k}, _t{k}:', self._c[k].detach().cpu().numpy(), self._t[k].detach().cpu().numpy())
        print(self.opt.cd_weight)
        print()

    def _eval_losses(self, render_pkg, viewpoint_cam, gaussians, gt_gaussians=None, requires_cd=False, i=None):
        gt_image = viewpoint_cam.original_image.cuda()
        losses = {
            'im': eval_img_loss(render_pkg['render'], gt_image, self.opt),
            'bce': None,
            'd': None,
            'center': None,
            'ppp': None,
            'ppped': None,
            'kl': None,
            'sd': None,
        }
        loss = losses['im']
        if (self.opt.sd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            mp_indices = self.pred_mp()
            pc_lst = []
            num = gaussians.size() // (1 + self.num_movable)
            pc_lst += [
                gaussians.get_xyz[num * (k + 1) : num * (k + 2)][
                    (self.get_prob > self.opt.mask_thresh) & (mp_indices == k)
                ] for k in np.arange(self.num_movable)
            ]
            pc_lst.append(gaussians.get_xyz[:num][self.get_prob < (1 - self.opt.mask_thresh)])
            x = sample_pts(torch.cat(pc_lst, dim=0), 5000)
            y = sample_pts(gt_gaussians.get_xyz, 5000)
            losses['sd'] = self.loss_fn(x.unsqueeze(0), y.unsqueeze(0))[0]
            loss += self.opt.sd_weight * losses['sd']

            # pcd_deformed = torch.cat(self.pcds_deformed, dim=0)
            # pred = sample_pts(pcd_deformed, 5000)
            # gt = sample_pts(self.pcd_gt, -1)
            # # losses['sd'] = self.loss_fn(pred.unsqueeze(0), gt.unsqueeze(0))[0]
            # losses['sd'] = self.loss_fn(gt.unsqueeze(0), pred.unsqueeze(0))[0]
            # loss += self.opt.sd_weight * losses['sd']

            # st_movable_mask = self.get_prob > self.opt.mask_thresh
            # ed_movable_mask = ~(~st_movable_mask)[self.ed_knn_indices]
            # mp_indices = self.pred_mp()
            # num = gaussians.size() // (1 + self.num_movable)
            # pc_lst = []
            # pc_lst += [
            #     gaussians.get_xyz[num * (k + 1) : num * (k + 2)][st_movable_mask & (mp_indices == k)]
            #     for k in np.arange(self.num_movable)
            # ]
            # x = sample_pts(torch.cat(pc_lst, dim=0), 5000)
            # y = sample_pts(gt_gaussians.get_xyz[ed_movable_mask], 5000)
            # losses['sd'] = self.loss_fn(x.unsqueeze(0), y.unsqueeze(0))[0]
            # loss += self.opt.sd_weight * losses['sd']
            # if i == 2000:
            #     gt_gaussians[ed_movable_mask].save_ply(
            #         os.path.join(self.dataset.model_path, f'point_cloud/iteration_{-2}/point_cloud.ply'),
            #     )

        if (self.opt.cd_weight is not None) and (gt_gaussians is not None) and requires_cd:
            # mp_indices = self.pred_mp()
            # pc_lst = []
            # num = gaussians.size() // (1 + self.num_movable)
            # pc_lst += [
            #     gaussians.get_xyz[num * (k + 1) : num * (k + 2)][
            #         (self.get_prob > self.opt.mask_thresh) & (mp_indices == k)
            #     ] for k in np.arange(self.num_movable)
            # ]
            # pc_lst.append(gaussians.get_xyz[:num][self.get_prob < (1 - self.opt.mask_thresh)])
            # x = sample_pts(torch.cat(pc_lst, dim=0), 5000)
            # y = sample_pts(gt_gaussians.get_xyz, 5000)
            # dist1, _ = chamfer_distance(x.unsqueeze(0), y.unsqueeze(0), batch_reduction=None)
            # # dist2, _ = chamfer_distance(y.unsqueeze(0), x.unsqueeze(0), batch_reduction=None)
            # # losses['cd'] = (dist1[0] + dist2[0]) * 0.5
            # losses['cd'] = dist1[0]
            # loss += self.opt.cd_weight * losses['cd']

            pcd_deformed = torch.cat(self.pcds_deformed, dim=0)
            x = sample_pts(pcd_deformed, 5000)
            y = sample_pts(self.pcd_gt, -1)
            dist, _ = chamfer_distance(x.unsqueeze(0), y.unsqueeze(0), batch_reduction=None)
            # dist, _ = chamfer_distance(y.unsqueeze(0), x.unsqueeze(0), batch_reduction=None)
            losses['cd'] = dist[0]
            loss += self.opt.cd_weight * losses['cd']

            # st_movable_mask = self.get_prob > self.opt.mask_thresh
            # ed_movable_mask = ~(~st_movable_mask)[self.ed_knn_indices]
            # mp_indices = self.pred_mp()
            # num = gaussians.size() // (1 + self.num_movable)
            # pc_lst = []
            # pc_lst += [
            #     gaussians.get_xyz[num * (k + 1) : num * (k + 2)][st_movable_mask & (mp_indices == k)]
            #     for k in np.arange(self.num_movable)
            # ]
            # x = sample_pts(torch.cat(pc_lst, dim=0), 5000)
            # y = sample_pts(gt_gaussians.get_xyz[ed_movable_mask], 5000)
            # # dist, _ = chamfer_distance(x.unsqueeze(0), y.unsqueeze(0), batch_reduction=None)
            # dist, _ = chamfer_distance(y.unsqueeze(0), x.unsqueeze(0), batch_reduction=None)
            # losses['cd'] = dist[0]
            # loss += self.opt.cd_weight * losses['cd']
            # if i == 2000:
            #     gt_gaussians[ed_movable_mask].save_ply(
            #         os.path.join(self.dataset.model_path, f'point_cloud/iteration_{-2}/point_cloud.ply'),
            #     )

        if (self.opt.depth_weight is not None) and (viewpoint_cam.image_depth is not None):
            gt_depth = viewpoint_cam.image_depth.cuda()
            losses['d'] = eval_depth_loss(render_pkg['depth'], gt_depth, scaling=self.opt.depth_scaling)
            loss += self.opt.depth_weight * losses['d']

        if self.opt.center_weight is not None and self.num_movable > 1:
            mask = self.get_ppp() * self.get_prob.unsqueeze(-1) # [N, K]
            c = torch.einsum('nk,nj->kj', mask, self.original_xyz.to(dtype=mask.dtype))  # [K, 3]
            c /= mask.sum(dim=0).unsqueeze(-1)
            losses['center'] = nn.functional.mse_loss(self.get_mu, c)
            loss += self.opt.center_weight * losses['center']

        if self.opt.ppp_weight is not None and self.num_movable > 1:
            losses['ppp'] = 0
            for k in range(self.num_movable):
                losses['ppp'] += torch.cat(
                    [self.get_ppp()[self.pcd_knn_indices[i]] for i in range(self.num_movable) if i != k], dim=0
                )[:, k].mean()
            loss += self.opt.ppp_weight * losses['ppp'] / self.num_movable

        if self.opt.ppp_weight_ed is not None and self.num_movable > 1 and requires_cd:
            losses['ppped'] = 0
            for k in range(self.num_movable):
                losses['ppped'] += self.get_ppp(
                    pts=torch.cat(
                        [
                            torch.matmul(self.pcds_sample[i], self.get_r[i]) + self.get_t[i]
                            for i in range(self.num_movable) if i != k
                        ], dim=0
                    ), deformed=True
                )[:, k].mean()
            loss += self.opt.ppp_weight_ed * losses['ppped'] / self.num_movable

        if self.opt.kl_weight is not None and self.num_movable > 1 and requires_cd:
            r = torch.stack(self.get_r).to(dtype=self.get_mu.dtype)
            t = torch.stack(self.get_t).to(dtype=self.get_mu.dtype)
            rot = r.transpose(1, 2) @ self.get_rotation
            ss = torch.diag_embed(self.get_scaling)
            sigma = rot @ ss @ ss @ rot.transpose(1, 2)
            mu = torch.einsum('kji,kj->ki', r, self.get_mu) + t
            losses['kl'] = 0
            for a, m in zip(mu, sigma):
                for aa, mm in zip(mu, sigma):
                    losses['kl'] += kl_divergence_gaussian(a, m, aa, mm)
            losses['kl'] /= self.num_movable
            loss += self.opt.kl_weight * losses['kl']

        return loss, losses

    def _set_ed_knn_indices(self, gt_gaussians: GaussianModel):
        p1 = gt_gaussians.get_xyz.detach().unsqueeze(0)
        p2 = self.original_xyz.detach().unsqueeze(0)
        _, indices, _ = knn_points(p1, p2, K=1)
        self.ed_knn_indices = indices.flatten()

    @override
    def train(self, gt_gaussians):
        _ = prepare_output_and_logger(self.dataset)
        iterations = self.opt.iterations
        bws = BWScenes(self.dataset, self.gaussians, is_new_gaussians=False)
        self.training_setup(self.opt)
        # self.joint_pred()
        self._set_ed_knn_indices(gt_gaussians)

        progress_bar = tqdm(range(iterations), desc="Training progress")
        ema_loss_for_log = 0.0
        for i in range(1, iterations + 1):
            requires_cd = self.opt.cd_from_iter <= i <= self.opt.cd_until_iter
            if self.opt.cd_weight is not None:
                self.opt.cd_weight = self.cosine_anneal(
                    i, final_step=self.opt.cd_until_iter, start_step=self.opt.cd_from_iter,
                    start_value=self.opt.cd_from_weight, final_value=self.opt.cd_until_weight
                )

            # self._p = self.cosine_anneal(i, start_step=0, final_step=self.opt.warmup_until_iter, start_value=1, final_value=2)
            # self._p = self.cosine_anneal(i, start_step=0, final_step=self.opt.iterations, start_value=1, final_value=2)
            self.deform(i)

            # Pick a random Camera
            viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background)
            # render_pkg = render(viewpoint_cam, self.gaussians[self.gaussians.get_opacity.squeeze(-1) > 0.005], self.pipe, background)
            loss, losses = self._eval_losses(render_pkg, viewpoint_cam, self.gaussians, gt_gaussians, requires_cd=requires_cd, i=i)
            loss.backward()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if i % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)

                if i < iterations:
                    if i % self.opt.sgd_interval == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self._prob[:] = torch.clamp(self._prob, -16, 16)
                        self._opacity[:] = torch.clamp(self._opacity, -16, 16)
                        self._scaling[:] = torch.clamp(self._scaling, -16, 16)
                        # for t in self._t:
                        #     t[0::2] = 0.0
                    self.gaussians.get_opacity_raw = self.gaussians.get_opacity_raw.detach()
                    self.gaussians.get_xyz = self.gaussians.get_xyz.detach()
                    self.gaussians.get_rotation_raw = self.gaussians.get_rotation_raw.detach()
                    self.ppp = None

                if i == self.opt.warmup_until_iter:
                    print('')
                    for k in np.arange(self.num_movable):
                        self.is_revolute[k] = (torch.trace(self.get_r[k]) < self.opt.trace_r_thresh)
                        print(f'Detected part{k} is ' + ('REVOLUTE' if self.is_revolute[k] else 'PRISMATIC'))
                        if self.is_revolute[k]:
                            continue
                        # self._column_vec1[k] = nn.Parameter(
                        #     torch.tensor([1, 0, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        # )
                        # self._column_vec2[k] = nn.Parameter(
                        #     torch.tensor([0, 1, 0], dtype=torch.float, device='cuda').requires_grad_(False)
                        # )
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
    def deform(self, iteration):
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

            self.deform(i)

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
                # losses['collision'] = 0
                # for k in range(self.num_movable):
                #     msk = self.mask & (self.part_indices == k)
                #     losses['collision'] = eval_knn_opacities_collision_loss(self.gaussians, msk, k=self.opt.collision_knn)
                    # losses['collision'] += eval_knn_opacities_collision_loss(self.canonical_gaussians, msk, k=self.opt.collision_knn)
                ## ↑↑ ?????
                losses['collision'] = eval_knn_opacities_collision_loss(self.gaussians, self.mask, k=self.opt.collision_knn)
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
