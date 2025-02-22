import os
import torch
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import json
import numpy as np
import matplotlib.pyplot as plt
from train import prepare_output_and_logger
from arguments import get_default_args
from utils.loss_utils import eval_losses, show_losses, eval_img_loss, eval_opacity_bce_loss, eval_depth_loss
from scene import BWScenes
from scene.gaussian_model import GaussianModel
from utils.general_utils import get_per_point_cd, otsu_with_peak_filtering, inverse_sigmoid

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
        if (depth_weight is not None) and (viewpoint_cam.image_depth is not None) and (i > opt.opacity_reset_interval):
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
                gaussians.save_ply(os.path.join(dataset.model_path, f'point_cloud/iteration_{i}/point_cloud.ply'))

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

def plot_hist(x: torch.Tensor, path: str, bins=100):
    plt.figure()
    plt.hist(x.detach().cpu().numpy(), bins=bins)
    plt.savefig(path)

def get_gt_motion_params(data_path: str):
    r = np.eye(3)
    t = np.zeros(3)

    with open(os.path.join(data_path, 'trans.json'), 'r') as json_file:
        trans = json.load(json_file)
    trans_info = trans['trans_info']

    if trans_info['type'] == 'translate':
        direction = np.array(trans_info['axis']['d'])
        distance = trans_info['translate']['r'] - trans_info['translate']['l']
        t = direction * distance
    elif trans_info['type'] == 'rotate':
        c = np.array(trans_info['axis']['o'])
        n = np.array(trans_info['axis']['d'])
        n /= np.linalg.norm(n)
        theta = (trans_info['rotate']['r'] - trans_info['rotate']['l']) / 180 * np.pi
        r = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * (n[:, np.newaxis] @ n[np.newaxis, :]) + np.sin(theta) * (
            np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        )
        t = c - r @ c
        r = r.T
    else:
        assert False
    print('t:', t)
    print('r:', r)
    return t, r

def get_gt_motion_params_mp(data_path: str, reverse=False):
    with open(os.path.join(data_path, 'trans.json'), 'r') as json_file:
        trans = json.load(json_file)
    trans_infos = trans['trans_info']
    for trans_info in trans_infos:
        r = np.eye(3)
        if trans_info['type'] == 'translate':
            direction = np.array(trans_info['axis']['d'])
            distance = trans_info['translate']['r'] - trans_info['translate']['l']
            t = direction * distance
        elif trans_info['type'] == 'rotate':
            c = np.array(trans_info['axis']['o'])
            n = np.array(trans_info['axis']['d'])
            n /= np.linalg.norm(n)
            theta = (trans_info['rotate']['r'] - trans_info['rotate']['l']) / 180 * np.pi
            r = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * (n[:, np.newaxis] @ n[np.newaxis, :]) + np.sin(theta) * (
                np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
            )
            t = c - r @ c
            r = r.T
        else:
            assert False
        print('t:', t if not reverse else -r.T @ t)
        print('r:', r if not reverse else r.T)

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
