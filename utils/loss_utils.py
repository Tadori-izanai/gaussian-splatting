#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from utils.general_utils import build_rotation, quat_mult, weighted_l2_loss_v2, weighted_l2_loss_v1
from scene.gaussian_model import GaussianModel
from pytorch3d.loss import chamfer_distance

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _sample_pts(pts: torch.Tensor, n_samples: int) -> torch.Tensor:
    n_pts = pts.shape[0]
    if n_samples > n_pts:
        return pts
    indices = torch.randperm(n_pts)[:n_samples]
    return pts[indices]

def eval_img_loss(image, gt_image, opt) -> torch.Tensor:
    ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    return loss

def eval_cd_loss(gaussians: GaussianModel, gt_gaussians: GaussianModel, n_samples=10000) -> torch.Tensor:
    pts = _sample_pts(gaussians.get_xyz, n_samples).unsqueeze(0)
    gt_pts = _sample_pts(gt_gaussians.get_xyz.detach(), n_samples).unsqueeze(0)
    dist1, _ = chamfer_distance(pts, gt_pts, batch_reduction=None)
    return dist1[0]

def eval_cd_loss_sd(gaussians: GaussianModel, gt_gaussians: GaussianModel) -> torch.Tensor:
    pts = gaussians.get_xyz.unsqueeze(0)
    gt_pts = gt_gaussians.get_xyz.unsqueeze(0)
    dist, _ = chamfer_distance(pts, gt_pts, batch_reduction=None)
    return dist[0]

def eval_rigid_loss(gaussians: GaussianModel) -> torch.Tensor:
    curr_rot = gaussians.get_rotation
    relative_rot = quat_mult(curr_rot, gaussians.prev_rotation_inv)
    rotation = build_rotation(relative_rot)

    prev_offset = gaussians.prev_xyz[gaussians.neighbor_indices] - gaussians.prev_xyz[:, None]
    curr_offset = gaussians.get_xyz[gaussians.neighbor_indices] - gaussians.get_xyz[:, None]
    curr_offset_in_prev_coord = (rotation.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
    return weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offset, gaussians.neighbor_weight)

def eval_rot_loss(gaussians: GaussianModel) -> torch.Tensor:
    curr_rot = gaussians.get_rotation
    relative_rot = quat_mult(curr_rot, gaussians.prev_rotation_inv)
    return weighted_l2_loss_v2(
        relative_rot[gaussians.neighbor_indices], relative_rot[:, None], gaussians.neighbor_weight
    )

def eval_iso_loss(gaussians: GaussianModel) -> torch.Tensor:
    curr_offset = gaussians.get_xyz[gaussians.neighbor_indices] - gaussians.get_xyz[:, None]
    curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
    return weighted_l2_loss_v1(curr_offset_mag, gaussians.neighbor_dist, gaussians.neighbor_weight)

def eval_opacity_bce_loss(op: torch.tensor):
    gt = (op > .5).float()
    return F.binary_cross_entropy(op, gt, reduction='mean')

def eval_losses(opt, iteration, image, gt_image, gaussians: GaussianModel=None, gt_gaussians: GaussianModel=None):
    losses = {
        'im': eval_img_loss(image, gt_image, opt),
        'rigid': None,
        'rot': None,
        'iso': None,
        'cd': None,
    }
    loss = losses['im']
    if opt.rigid_weight is not None:
        losses['rigid'] = eval_rigid_loss(gaussians)
        loss += opt.rigid_weight * losses['rigid']
    if opt.rot_weight is not None:
        losses['rot'] = eval_rot_loss(gaussians)
        loss += opt.rot_weight * losses['rot']
    if opt.iso_weight is not None:
        losses['iso'] = eval_iso_loss(gaussians)
        loss += opt.iso_weight * losses['iso']
    if (opt.cd_weight is not None) and (gt_gaussians is not None):
        if opt.cd_from_iter <= iteration <= opt.cd_until_iter:
            losses['cd'] = eval_cd_loss(gaussians, gt_gaussians, opt.cd_numbers)
            loss += opt.cd_weight * losses['cd']
    return loss, losses

def show_losses(iteration: int, losses: dict):
    if iteration in [10, 100, 1000, 5000-1, 10000, 20000, 30000]:
        loss_msg = "\n"
        for name, loss in losses.items():
            if loss is not None:
                loss_msg += f"  {name} {loss.item():.{7}f}"
        print(loss_msg)
