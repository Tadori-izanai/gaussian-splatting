import os
import torch
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import json
import numpy as np
import matplotlib.pyplot as plt
from train import prepare_output_and_logger
from arguments import get_default_args
from utils.loss_utils import eval_losses, show_losses, eval_img_loss, eval_opacity_bce_loss, eval_depth_loss
from scene import BWScenes
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readCamerasFromTransforms
from scene.multipart_models import GMMArtModel
from utils.general_utils import get_per_point_cd, otsu_with_peak_filtering, inverse_sigmoid, \
    decompose_covariance_matrix, rotation_matrix_from_axis_angle, eval_quad, knn
from utils.graphics_utils import getProjectionMatrix, getWorld2View

def train_single(dataset, opt, pipe, gaussians: GaussianModel, bce_weight=None, depth_weight=None):
    _ = prepare_output_and_logger(dataset)
    bws = BWScenes(dataset, gaussians, is_new_gaussians=True)
    gaussians.training_setup(opt)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for i in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(i)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if i % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        viewpoint_cam, background = bws.pop_black() if (i % 2 == 0) else bws.pop_white()

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        loss = eval_img_loss(image, gt_image, opt)

        if bce_weight is not None:
            loss += bce_weight * eval_opacity_bce_loss(gaussians.get_opacity)
        # if (depth_weight is not None) and (viewpoint_cam.image_depth is not None) and (i > opt.opacity_reset_interval):
        if (depth_weight is not None) and (viewpoint_cam.image_depth is not None):
            depth = render_pkg['depth']
            gt_depth = viewpoint_cam.image_depth.cuda()
            loss += depth_weight * eval_depth_loss(depth, gt_depth)

        loss.backward()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if i % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            # Save ply
            if i in [7000, 30000]:
                gaussians.save_ply(
                    os.path.join(dataset.model_path, f'point_cloud/iteration_{i}/point_cloud.ply'),
                    prune=True
                )

            # Densification
            if i < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # copy or split
                if i > opt.densify_from_iter and i % opt.densification_interval == 0:
                    size_threshold = 20 if i > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, bws.get_cameras_extent(), size_threshold)
                # opacity reset
                if i % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and i == opt.densify_from_iter):
                    gaussians.reset_opacity()
            # Optimizer step
            if i < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
    progress_bar.close()
    with torch.no_grad():
        prune_mask = (gaussians.get_opacity < 0.005).squeeze()
        gaussians.prune_points(prune_mask)
    torch.save((gaussians.capture(), opt.iterations), os.path.join(dataset.model_path, 'chkpnt.pth'))

def get_gaussians(model_path, from_chk=True, iters=30003) -> GaussianModel:
    gaussians = GaussianModel(0)
    if from_chk:
        dataset, pipes, opt = get_default_args()
        model_params, _ = torch.load(os.path.join(model_path, 'chkpnt.pth'))
        gaussians.restore(model_params, opt)
    else:
        gaussians.load_ply(os.path.join(model_path, f'point_cloud/iteration_{iters}/point_cloud.ply'))
    return gaussians

def print_motion_params(out_path: str):
    t = np.load(os.path.join(out_path, 't_pre.npy'))
    r = np.load(os.path.join(out_path, 'r_pre.npy'))
    print('t:', t)
    print('r:', r)

def plot_hist(x, path: str, bins=100):
    plt.figure()
    try:
        plt.hist(x, bins=bins)
    except:
        plt.hist(x.detach().cpu().numpy(), bins=bins)
    plt.savefig(path)

def mk_output_dir(out_path: str, data_path: str):
    os.makedirs(out_path, exist_ok=True)
    dataset, pipes, opt = get_default_args()
    dataset.eval = True
    dataset.sh_degree = 0
    dataset.source_path = os.path.realpath(data_path)
    dataset.model_path = out_path
    _ = prepare_output_and_logger(dataset)

def init_mpp(gaussians_st: GaussianModel, gaussians_ed: GaussianModel, thr=None, sig_scale=1.0):
    cds_st = get_per_point_cd(gaussians_st, gaussians_ed)
    cds_st_normalized = cds_st / torch.max(cds_st)

    eps = 1e-6
    csn_is = inverse_sigmoid(torch.clamp(cds_st_normalized, eps, 1-eps))
    if thr is None:
        thr = otsu_with_peak_filtering(csn_is.detach().cpu().numpy(), bias_factor=1.25)
        print(thr)
    csn_shifted = torch.sigmoid((inverse_sigmoid(cds_st_normalized) - thr) * sig_scale)
    return cds_st_normalized, csn_is, csn_shifted

# def get_cluster_centers(pts: torch.tensor, num: int) -> torch.tensor:
#     kmeans = KMeans(n_clusters=num, random_state=42)
#     kmeans.fit(pts.detach().cpu().numpy())
#     centroids = kmeans.cluster_centers_
#     return torch.tensor(centroids, device=pts.device)
#
# def get_ppp_from_dist(pts: torch.tensor, centers: torch.tensor):
#     dist_to_centers = (pts.unsqueeze(1) - centers).norm(dim=2)
#     prob = dist_to_centers / torch.sum(dist_to_centers, dim=1, keepdim=True)
#     return prob

def get_ppp_from_gmm(train_pts: torch.tensor, test_pts: torch.tensor, num: int) -> torch.tensor:
    gmm = GaussianMixture(n_components=num, random_state=42)
    gmm.fit(train_pts.detach().cpu().numpy())
    prob = gmm.predict_proba(test_pts.detach().cpu().numpy())
    return torch.tensor(prob, device=test_pts.device)

def get_ppp_from_gmm_v2(train_pts: torch.tensor, test_pts: torch.tensor, num: int) -> torch.tensor:
    gmm = GaussianMixture(n_components=num, random_state=42)
    gmm.fit(train_pts.detach().cpu().numpy())
    means = gmm.means_
    covariances = gmm.covariances_

    inv_covariances = np.linalg.inv(covariances)
    diff = test_pts.detach().cpu().numpy()[:, np.newaxis, :] - means
    quad = np.einsum('pki,kij,pkj->pk', diff, inv_covariances, diff)
    # prob = 1 / (np.sqrt(quad) + 1e-6)
    prob = 1 / (quad ** 2 + 1e-6)
    prob /= np.sum(prob, axis=1, keepdims=True)
    return torch.tensor(prob, device=test_pts.device)

def eval_init_gmm_params(train_pts: torch.tensor, num: int) -> tuple[torch.tensor, torch.tensor]:
    gmm = GaussianMixture(n_components=num, random_state=42)
    gmm.fit(train_pts.detach().cpu().numpy())
    means = torch.tensor(gmm.means_, device=train_pts.device)
    covariances = torch.tensor(gmm.covariances_, device=train_pts.device)
    return means, covariances

def eval_mu_sigma(pts: np.ndarray) -> tuple[np.array, np.array]:
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(pts)
    return gmm.means_, gmm.covariances_

def modify_scaling(cov: torch.tensor, scaling_modifier=1.0) -> torch.tensor:
    scaling, rotation = decompose_covariance_matrix(cov)
    scaling = torch.diag_embed(scaling * scaling_modifier)
    return rotation @ scaling @ scaling @ rotation.transpose(1, 2)

def get_depths(
    pts: np.ndarray,  # given pts in world coordinate
    depth_map: np.ndarray,
    R: np.array,
    T: np.array,
    FovY: np.array,
    FovX: np.array,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    zfar = 100.0
    znear = 0.01
    projection_matrix = getProjectionMatrix(znear, zfar, FovX, FovY, blender_convention=True).cpu().numpy()
    w2c_colmap = getWorld2View(R, T)
    c2w_colmap = np.linalg.inv(w2c_colmap)
    c2w_blender = c2w_colmap.copy()
    c2w_blender[:3, 1:3] *= -1
    w2c_blender = np.linalg.inv(c2w_blender)

    # Transform world coordinates to camera space
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N, 4)
    pts_camera = (w2c_blender @ pts_homo.T).T  # (N, 4)
    # Transform camera space to clip space
    pts_clip = (projection_matrix @ pts_camera.T).T  # (N, 4)
    pts_clip /= pts_clip[:, 3, np.newaxis]      # Perspective division
    pts_camera /= pts_camera[:, 3, np.newaxis]  # Perspective division
    # Convert to pixel coordinates
    x_pixel = ((pts_clip[:, 0] + 1) * image_width / 2).astype(int)
    y_pixel = ((1 - pts_clip[:, 1]) * image_height / 2).astype(int)
    # Clamp pixel coordinates to valid range
    x_pixel = np.clip(x_pixel, 0, image_width - 1)
    y_pixel = np.clip(y_pixel, 0, image_height - 1)
    # Get depths from the depth map
    pixel_depths = depth_map[y_pixel, x_pixel]

    pixel_depths[pixel_depths < 0.01] = 114.514
    return -pts_camera[:, 2], pixel_depths

def eval_visibility(pts: np.array, data_path: str, eps: float=0.01) -> np.array:
    data_path = os.path.realpath(data_path)
    train_cam_infos = readCamerasFromTransforms(data_path, "transforms_train.json", white_background=False)
    vis = np.zeros(len(pts), dtype='bool')
    for cam_info in train_cam_infos:
        depth_pts, depth_obj = get_depths(
            pts=pts,
            depth_map=cam_info.image_d,
            R=cam_info.R,
            T=cam_info.T,
            FovX=cam_info.FovX,
            FovY=cam_info.FovY,
            image_width=cam_info.width,
            image_height=cam_info.height,
        )
        vis |= (depth_pts < depth_obj + eps)
    return vis

def get_vis_mask(gaussians: GaussianModel, data_path: str, eps: float=0.01) -> torch.tensor:
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    return torch.tensor(eval_visibility(xyz, data_path, eps), device=gaussians.get_xyz.device)

def value_to_rgb(values, cmap_name='viridis'):
    """
    将 0~1 之间的值映射到 RGB 颜色。

    参数:
    values: (N,) 形状的 Tensor，值范围在 [0,1]。
    cmap_name: 字符串，可选，Matplotlib 的 colormap 名称。

    返回:
    (N, 3) 形状的 Tensor，RGB 颜色值范围在 [0,1]。
    """
    cmap = plt.get_cmap(cmap_name)
    values = values.detach().cpu().numpy()  # 转换为 NumPy 以使用 Matplotlib
    colors = cmap(values)[:, :3]   # 获取 RGB 值（忽略 alpha 通道）
    return torch.tensor(colors, dtype=torch.float32)

def estimate_se3(p: torch.tensor, p_prime: torch.tensor, k_neighbors=21):
    p = p.detach().cpu().numpy()
    p_prime = p_prime.detach().cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(p)
    _, indices = nbrs.kneighbors(p)  # indices: (n_samples, k_neighbors)

    def estimate_se3_batch(p_subset, p_prime_subset):
        # p_subset 和 p_prime_subset: (n_samples, k_neighbors, 3)
        # 计算质心
        p_mean = np.mean(p_subset, axis=1, keepdims=True)  # (n_samples, 1, 3)
        p_prime_mean = np.mean(p_prime_subset, axis=1, keepdims=True)
        # 去中心化
        p_centered = p_subset - p_mean  # (n_samples, k_neighbors, 3)
        p_prime_centered = p_prime_subset - p_prime_mean
        # 协方差矩阵 H
        H = np.einsum('ijk,ijl->ikl', p_centered, p_prime_centered)  # (n_samples, 3, 3)
        # SVD 分解
        U, _, Vt = np.linalg.svd(H)  # U, Vt: (n_samples, 3, 3)
        R = np.einsum('ijk,ikl->ijl', Vt, U)  # (n_samples, 3, 3)
        # 修正旋转矩阵（确保 det(R) = 1）
        det_R = np.linalg.det(R)
        Vt[:, :, -1] *= np.sign(det_R)[:, None]  # 调整最后一行
        R = np.einsum('ijk,ikl->ijl', Vt, U)
        # 计算平移
        t = p_prime_mean.squeeze(axis=1) - np.einsum('ijk,ik->ij', R, p_mean.squeeze(axis=1))
        return R, t

    # 获取邻域点集
    p_neighbors = p[indices]  # (n_samples, k_neighbors, 3)
    p_prime_neighbors = p_prime[indices]
    return estimate_se3_batch(p_neighbors, p_prime_neighbors)  # R: (n_samples, 3, 3), t: (n_samples, 3)

if __name__ == '__main__':
    def init_demo_v2(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int):
        mk_output_dir(out_path, os.path.join(data_path, 'start'))
        gaussians_st = get_gaussians(st_path, from_chk=True)
        gaussians_ed = get_gaussians(ed_path, from_chk=True)

        st_data = os.path.join(data_path, 'start')
        ed_data = os.path.join(data_path, 'end')
        st_mask = get_vis_mask(gaussians_st, ed_data)
        ed_mask = get_vis_mask(gaussians_ed, st_data)
        gaussians_st = gaussians_st[st_mask]
        gaussians_ed = gaussians_ed[ed_mask]

        cd, cd_is, mpp = init_mpp(gaussians_st, gaussians_ed, thr=-4)
        mask_s = (mpp < .5)
        mu, sigma = eval_init_gmm_params(train_pts=gaussians_st[~mask_s].get_xyz, num=num_movable)

        gaussians_st[~mask_s].save_ply(os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
        np.save(os.path.join(out_path, 'mu_init.npy'), mu.detach().cpu().numpy())
        np.save(os.path.join(out_path, 'sigma_init.npy'), sigma.detach().cpu().numpy())

        ## gmm_am_optim_demo_v2
        torch.autograd.set_detect_anomaly(False)
        gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
        am = GMMArtModel(gaussians_st, num_movable)
        am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'), model_path=out_path)
        am.set_init_params(out_path, scaling_modifier=1)
        am.save_ppp_vis(os.path.join(out_path, 'point_cloud/iteration_9/point_cloud.ply'))

    K = 6
    st = 'output/sto6_st'
    ed = 'output/sto6_ed'
    data = 'data/artgs/storage_47648'
    out = 'output/sto6'
    rev = True

    # K = 4
    # st = 'output/tbr4_st'
    # ed = 'output/tbr4_ed'
    # data = 'data/teeburu34178'
    # out = 'output/tbr4'
    # rev = False

    init_demo_v2(out, st, ed, data, num_movable=K)

    pass
