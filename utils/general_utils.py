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
import sys
from datetime import datetime
import numpy as np
import random
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

from pytorch3d.loss import chamfer_distance

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def knn(points: np.ndarray, num_knn: int):
    tree = KDTree(points)
    distances, indices = tree.query(points, k=num_knn + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    return distances, indices

def quat_mult(q1, q2):
    # w1, x1, y1, z1 = q1.T
    # w2, x2, y2, z2 = q2.T
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

def mat2quat(m):
    t = torch.trace(m)
    if t > 0:
        s = torch.sqrt(1.0 + t) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return torch.tensor([w, x, y, z])

def mat2quat_batch(m):
    # 计算 trace，得到的 t 是 (B,)
    t = torch.sum(torch.diagonal(m, dim1=-2, dim2=-1), dim=-1)

    # 初始化 w, x, y, z 为零
    w = torch.zeros(m.shape[0], device=m.device)
    x = torch.zeros(m.shape[0], device=m.device)
    y = torch.zeros(m.shape[0], device=m.device)
    z = torch.zeros(m.shape[0], device=m.device)

    # 如果 t > 0
    mask = t > 0
    s = torch.sqrt(1.0 + t) * 2
    w[mask] = 0.25 * s[mask]
    x[mask] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    y[mask] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    z[mask] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    # 如果 m[0, 0] > m[1, 1] 和 m[0, 0] > m[2, 2]
    mask_1 = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s_1 = torch.sqrt(1.0 + m[mask_1, 0, 0] - m[mask_1, 1, 1] - m[mask_1, 2, 2]) * 2
    w[mask_1] = (m[mask_1, 2, 1] - m[mask_1, 1, 2]) / s_1
    x[mask_1] = 0.25 * s_1
    y[mask_1] = (m[mask_1, 0, 1] + m[mask_1, 1, 0]) / s_1
    z[mask_1] = (m[mask_1, 0, 2] + m[mask_1, 2, 0]) / s_1

    # 如果 m[1, 1] > m[2, 2]
    mask_2 = m[:, 1, 1] > m[:, 2, 2]
    s_2 = torch.sqrt(1.0 + m[mask_2, 1, 1] - m[mask_2, 0, 0] - m[mask_2, 2, 2]) * 2
    w[mask_2] = (m[mask_2, 0, 2] - m[mask_2, 2, 0]) / s_2
    x[mask_2] = (m[mask_2, 0, 1] + m[mask_2, 1, 0]) / s_2
    y[mask_2] = 0.25 * s_2
    z[mask_2] = (m[mask_2, 1, 2] + m[mask_2, 2, 1]) / s_2

    # 如果 m[2, 2] 是最大元素
    mask_3 = ~mask & ~mask_1 & ~mask_2
    s_3 = torch.sqrt(1.0 + m[mask_3, 2, 2] - m[mask_3, 0, 0] - m[mask_3, 1, 1]) * 2
    w[mask_3] = (m[mask_3, 1, 0] - m[mask_3, 0, 1]) / s_3
    x[mask_3] = (m[mask_3, 0, 2] + m[mask_3, 2, 0]) / s_3
    y[mask_3] = (m[mask_3, 1, 2] + m[mask_3, 2, 1]) / s_3
    z[mask_3] = 0.25 * s_3

    # 返回四元数 [w, x, y, z]
    return torch.stack([w, x, y, z], dim=1)

def otsu_with_peak_filtering(data, std_multiplier=3, bins=256, bias_factor=1.25):
    """
    使用主峰截取后再应用 Otsu 方法计算阈值。

    参数:
        data (numpy.ndarray): 输入数据，1D 数组。
        std_multiplier (float): 主峰范围的标准差倍数，默认值为3。
        bins (int): 用于 Otsu 方法的直方图分箱数，默认值为256。

    返回:
        float: 计算得到的阈值。
    """
    # 标准化
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)

    # 密度估计
    density = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)
    y_vals = density(x_vals)

    # 找到主峰位置
    peak_idx = np.argmax(y_vals)
    peak_value = x_vals[peak_idx]

    # 设定主峰附近的截取范围
    std_dev = np.std(data)
    lower_bound = peak_value - std_multiplier * std_dev
    upper_bound = peak_value + std_multiplier * std_dev
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    # Otsu 方法计算阈值
    def otsu_threshold_bias_towards_higher(data, bins=256, bias_factor=1.25):
        """
        修改版 Otsu 方法，倾向于输出较大的阈值。

        参数:
            data (numpy.ndarray): 输入数据，1D 数组。
            bins (int): 用于直方图分箱数，默认值为256。
            bias_factor (float): 偏向大阈值的因子（>1 时倾向更大值）。

        返回:
            float: 计算得到的阈值。
        """
        # 计算直方图
        hist, bin_edges = np.histogram(data, bins=bins, range=(min(data), max(data)), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total_weight = np.sum(hist)
        total_mean = np.sum(hist * bin_centers)
        max_between_class_variance = 0
        threshold = 0

        weight_bg = 0
        mean_bg = 0

        for i in range(len(hist)):
            weight_bg += hist[i]
            mean_bg += hist[i] * bin_centers[i]

            weight_fg = total_weight - weight_bg
            if weight_bg <= 0 or weight_fg <= 0:
                continue

            mean_fg = (total_mean - mean_bg) / weight_fg
            # between_class_variance = weight_bg * weight_fg * (mean_bg / weight_bg - mean_fg) ** 2
            # 缩小两峰大小之间的差距
            between_class_variance = weight_bg ** .5 * weight_fg ** .5 * (mean_bg / weight_bg - mean_fg) ** 2

            # 增加偏向大阈值的因子
            between_class_variance *= bin_centers[i] ** (bias_factor - 1)

            if between_class_variance > max_between_class_variance:
                max_between_class_variance = between_class_variance
                threshold = bin_centers[i]

        return threshold

    threshold = otsu_threshold_bias_towards_higher(filtered_data, bins, bias_factor=bias_factor)
    # 反标准化化
    threshold = min_val + (max_val - min_val) * threshold
    return threshold

def get_per_point_cd(gaussians_x, gaussians_y) -> torch.Tensor:
    x = gaussians_x.get_xyz.unsqueeze(0)
    y = gaussians_y.get_xyz.unsqueeze(0)
    dist, _ = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, single_directional=True)
    return dist[0]
