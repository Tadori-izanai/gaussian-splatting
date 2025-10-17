import os
import torch

from scene import Scene, GaussianModel
from utils.general_utils import safe_state, pca_on_pointcloud, \
    calculate_obb_o3d, calculate_obb_pv, estimate_normals_o3d, estimate_normals_pv, get_oriented_aabb, \
    get_bounding_box, estimate_normals_ransac_o3d, DisjointSet, get_rotation_axis

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import json
import numpy as np
from pytorch3d.loss import chamfer_distance
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from arguments import get_default_args
from utils.general_utils import eval_quad, inverse_sigmoid, value_to_rgb, estimate_principal_directions, \
    find_and_unify_orthogonal
from scene.multipart_models import MPArtModelJoint, GMMArtModel, COLORS
from scene.dataset_readers import fetchPly, storePly

from main_utils import train_single, get_gaussians, print_motion_params, plot_hist, \
    mk_output_dir, init_mpp, get_ppp_from_gmm, get_ppp_from_gmm_v2, eval_init_gmm_params, \
    modify_scaling, get_vis_mask, estimate_se3, eval_mu_sigma, save_axis_mesh, list_of_clusters, \
    put_axes, get_obb_proximity_matrix, get_tr_proximity_matrix, get_minimum_angles
from metric_utils import get_gt_motion_params, interpret_transforms, eval_axis_metrics, \
    get_pred_point_cloud, get_gt_point_clouds, eval_geo_metrics, stat_axis_metrics

cd_thr = -5
dbs_eps = 0.008
from_pgsr = False

def train_single_demo(path, data_path):
    dataset, pipes, opt = get_default_args()
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    dataset.eval = True
    dataset.sh_degree = 0
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.source_path = os.path.realpath(data_path)
    dataset.model_path = path
    train_single(dataset, opt, pipes, gaussians, depth_weight=1.0, bce_weight=0.01)

def cluster_demo(out_path: str, data_path: str, num_movable: int, thr: int=-5):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    ply_path = os.path.join(out_path, 'clustering')
    os.makedirs(ply_path, exist_ok=True)
    os.makedirs(os.path.join(ply_path, 'clusters'), exist_ok=True)

    st_data = os.path.join(os.path.realpath(data_path), 'start')
    ed_data = os.path.join(os.path.realpath(data_path), 'end')
    xyz_st = np.asarray(fetchPly(os.path.join(st_data, 'points3d.ply')).points)
    xyz_ed = np.asarray(fetchPly(os.path.join(ed_data, 'points3d.ply')).points)

    x = torch.tensor(xyz_st, device='cuda').unsqueeze(0)
    y = torch.tensor(xyz_ed, device='cuda').unsqueeze(0)
    cd = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, single_directional=True)[0][0]
    cd /= torch.max(cd)
    cd_is = inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6))
    plot_hist(cd_is, os.path.join(ply_path, 'cd_is-1m.png'))

    mask = (cd_is > thr)
    x = x[0][mask].detach().cpu().numpy()

    if True:
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(x)
        distances, _ = neigh.kneighbors(x)
        distances = np.sort(distances[:, -1])
        plot_hist(distances, os.path.join(ply_path, 'dist-1m.png'))

    clustering = DBSCAN(eps=dbs_eps, min_samples=num_movable).fit(x)

    labels = clustering.labels_
    pts = sorted(
        [(k, x[labels == k]) for k in np.unique_values(labels)],
        key=lambda item: len(item[1]), reverse=True
    )
    for k, pcd in pts[:num_movable]:
        if k == -1:
            k, pcd = pts[num_movable]
            print('warning: has -1')
        normals = estimate_normals_o3d(pcd)
        storePly(os.path.join(ply_path, f'clusters/points3d_{k}.ply'), pcd, np.zeros_like(pcd), normals)
    storePly(os.path.join(ply_path, f'points3d.ply'), x, np.zeros_like(x))

def part_init_demo(out_path, st_path, ed_path, num_movable: int):
    gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr)
    gaussians_ed = get_gaussians(ed_path, from_chk=True, from_pgsr=from_pgsr)
    cd, cd_is, mpp = init_mpp(gaussians_st, gaussians_ed, thr=-4.5)
    mask_m = (mpp > .5)
    gaussians_st[mask_m].save_ply(
        os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
    plot_hist(mpp, os.path.join(out_path, 'mpp.png'))
    np.save(os.path.join(out_path, 'mpp_init.npy'), mpp.detach().cpu().numpy())

    pts = list_of_clusters(os.path.join(out_path, 'clustering/clusters'), num_movable)
    mu = np.zeros((num_movable, 3))
    sigma = np.zeros((num_movable, 3, 3))
    for i in np.arange(num_movable):
        mu[i], sigma[i] = eval_mu_sigma(pts[i])
    np.save(os.path.join(out_path, 'mu_init.npy'), mu)
    np.save(os.path.join(out_path, 'sigma_init.npy'), sigma)

def joint_init_demo(out_path: str, st_path: str, num_movable: int):
    def rotate_axes(axes_to_rotate: np.ndarray, theta_deg: float) -> np.ndarray:
        random_axis = np.random.rand(3)
        while np.linalg.norm(random_axis) < 1e-6:
            random_axis = np.random.rand(3)
        random_axis /= np.linalg.norm(random_axis)

        theta_rad = np.deg2rad(theta_deg)
        x, y, z = random_axis
        K = np.array([[0, -z, y],
                      [z, 0, -x],
                      [-y, x, 0]])
        I = np.eye(3)
        R = I + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * np.dot(K, K)
        return axes_to_rotate @ R.T

    nrm_dir = os.path.join(out_path, 'clustering/nrm_axes')
    gaussians_dir = os.path.join(out_path, 'clustering/axes_gaussians')
    os.makedirs(nrm_dir, exist_ok=True)
    os.makedirs(gaussians_dir, exist_ok=True)

    parts, nms = list_of_clusters(os.path.join(out_path, 'clustering/clusters'), num_movable, ret_normal=True)
    mu = np.load(os.path.join(out_path, 'mu_init.npy'))

    axes = np.zeros((num_movable, 3, 3))
    for k, nm in enumerate(nms):
        axes[k] = estimate_principal_directions(nm, ort='gs')

    # for k in [0, 1]:
    #     axes[k] = rotate_axes(axes[k], 5)

    neighbors = find_and_unify_orthogonal(axes)

    bb_centers, bb_extents = [], []
    for k, pcd in enumerate(parts):
        o = np.zeros(3)
        # dirs = dirs_p if (types[k] == 'p') else estimate_principal_directions(nms[k], ort='gs')
        dirs = axes[k]
        for i in np.arange(3):
            d = -dirs[i] if (dirs[i] @ mu[k] < 0) else dirs[i]
            save_axis_mesh(d, o, os.path.join(nrm_dir, f'axis{k}_{i}.ply'), mu[k])
            save_axis_mesh(d, o, os.path.join(gaussians_dir, f'axis{k}_{i}.ply'), mu[k],
                           to_gaussians=True, c=COLORS[i])
        # axes.append(dirs)
        centers, extents = get_bounding_box(pcd, dirs)
        bb_centers.append(centers)
        bb_extents.append(extents)
        print('done with axis', k)
    np.save(os.path.join(out_path, f'clustering/axes.npy'), axes)
    np.save(os.path.join(out_path, f'clustering/bb_centers.npy'), bb_centers)
    np.save(os.path.join(out_path, f'clustering/bb_extents.npy'), bb_extents)
    np.save(os.path.join(out_path, f'clustering/neighbors.npy'), neighbors)

    put_axes(out_path, st_path, num_movable)

def art_optim_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int, thr=0.85):
    torch.autograd.set_detect_anomaly(False)
    gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr).cancel_grads()
    gaussians_ed = get_gaussians(ed_path, from_chk=True, from_pgsr=from_pgsr).cancel_grads()

    am = GMMArtModel(gaussians_st, num_movable, new_scheme=False)
    am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'), model_path=out_path, thr=cd_thr)
    am.set_init_params(out_path, scaling_modifier=1)
    # am.set_init_params(out_path, scaling_modifier=1, use_priors=True)
    am.save_all_vis(-10)
    t, r = am.train(gt_gaussians=gaussians_ed)

    ppp = am.get_ppp().detach().cpu().numpy()
    part_indices = np.argmax(ppp, axis=1)
    mpp = am.get_prob.detach().cpu().numpy()
    mask = (mpp > thr)

    gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr).cancel_grads()
    am.save_all_vis(-20)
    for i in range(num_movable):
        gaussians_st[mask & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{21 + i}/point_cloud.ply')
        )
    gaussians_st[~mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_20/point_cloud.ply'))
    np.save(os.path.join(out_path, 'r_pre.npy'), [rr.detach().cpu().numpy() for rr in r])
    np.save(os.path.join(out_path, 't_pre.npy'), [tt.detach().cpu().numpy() for tt in t])
    np.save(os.path.join(out_path, 'mask_pre.npy'), mask)
    np.save(os.path.join(out_path, 'part_indices_pre'), part_indices)

def merge_subparts(out_path: str, num_movable: int) -> int:
    part_indices = np.load(os.path.join(out_path, 'part_indices_pre.npy'))
    r = np.load(os.path.join(out_path, 'r_pre.npy'))
    t = np.load(os.path.join(out_path, 't_pre.npy'))
    axes = np.load(os.path.join(out_path, 'clustering/axes.npy'))
    bb_centers = np.load(os.path.join(out_path, 'clustering/bb_centers.npy'))
    bb_extents = np.load(os.path.join(out_path, 'clustering/bb_extents.npy'))

    volume_ratios, _ = get_obb_proximity_matrix(axes, bb_centers, bb_extents * 1.05)
    tr_close = get_tr_proximity_matrix(r, t)
    min_angles = get_minimum_angles(axes, r, t)
    print(volume_ratios)
    print(tr_close)
    print(min_angles)

    ds = DisjointSet(num_movable)
    for k in range(num_movable):
        if min_angles[k] > 2:
            j = np.argmax(volume_ratios[k])
            if min_angles[j] < 2:
                ds.connect(k, j)
                print('merge:', k, j)
            continue

        j, obb_max = -1, 0.04
        for i in range(num_movable):
            if i == k or not tr_close[k][i] or ds.is_connected(k, i):
                continue
            if volume_ratios[k][i] > obb_max:
                j, obb_max = i, volume_ratios[k][i]

        if j != -1:
            if min_angles[j] < min_angles[k]:
                ds.connect(k, j)
            else:
                ds.connect(j, k)
            print('merge:', k, j)

    merge_indices, uniques = ds.get_new_indices()
    new_part_indices = np.zeros_like(part_indices, dtype=int)
    for k in range(num_movable):
        new_part_indices[part_indices == k] = merge_indices[ds.parent[k]]
    new_r, new_t = [], []
    for idx in uniques:
        new_r.append(r[idx])
        new_t.append(t[idx])

    np.save(os.path.join(out_path, '_r_pre.npy'), new_r)
    np.save(os.path.join(out_path, '_t_pre.npy'), new_t)
    np.save(os.path.join(out_path, '_part_indices_pre'), new_part_indices)
    return len(uniques)

def vis_axes_pp_demo(out_path: str, st_path: str):
    ply_path = os.path.join(out_path, 'ply')
    os.makedirs(ply_path, exist_ok=True)
    # axes
    with open(os.path.join(out_path, 'trans_pred.json'), 'r') as json_file:
        trans = json.load(json_file)
    mu = np.load(os.path.join(out_path, 'mu_init.npy'))
    for i, trans_info in enumerate(trans):
        o = np.array(trans_info['axis']['o'])
        d = np.array(trans_info['axis']['d'])
        save_axis_mesh(d, o, os.path.join(ply_path, f'axis_{i}.ply'), mu[i])

    # pcd seg
    num_movable = len(mu)
    mask = np.load(os.path.join(out_path, 'mask_pre.npy'))
    part_indices = np.load(os.path.join(out_path, 'part_indices_pre.npy'))
    gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr).cancel_grads()
    xyz = gaussians_st.get_xyz.detach().cpu().numpy()
    rgb = np.full(xyz.shape, 255)
    for k in np.arange(num_movable):
        rgb[(part_indices == k) & mask] = np.array(COLORS[k % len(COLORS)]) * 255
    storePly(os.path.join(ply_path, 'seg.ply'), xyz, rgb)

def refinement_demo(out_path: str, st_path: str, data_path: str, num_movable: int):
    torch.autograd.set_detect_anomaly(False)

    gaussians_st = get_gaussians(st_path, from_chk=True, from_pgsr=from_pgsr)
    amj = MPArtModelJoint(gaussians_st, num_movable)
    amj.set_dataset(source_path=os.path.realpath(data_path), model_path=out_path)
    t, r = amj.train()

    mask, part_indices = amj.canonical_gaussians.save_ply(
        os.path.join(out_path, f'point_cloud/iteration_99999/point_cloud.ply'),
        prune=True, auxiliary_attr=(amj.mask, amj.part_indices)
    )
    gaussians_canonical = get_gaussians(out_path, from_chk=False, iters=99999)
    for i in range(num_movable):
        gaussians_canonical[mask & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{31 + i}/point_cloud.ply')
        )
    gaussians_canonical[~mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_30/point_cloud.ply'))
    gaussians_canonical[part_indices == 0].save_ply(os.path.join(out_path, 'point_cloud/iteration_29/point_cloud.ply'))
    np.save(os.path.join(out_path, 't_final.npy'), [tt.detach().cpu().numpy() for tt in t])
    np.save(os.path.join(out_path, 'r_final.npy'), [rr.detach().cpu().numpy() for rr in r])
    np.save(os.path.join(out_path, 'mask_final.npy'), mask.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'part_indices_final'), part_indices.detach().cpu().numpy())

def eval_demo(out_path: str, data_path: str, num_movable: int, reverse=True):
    t = np.load(os.path.join(out_path, 't_final.npy'))
    r = np.load(os.path.join(out_path, 'r_final.npy'))
    # t = np.load(os.path.join(out_path, 't_pre.npy'))
    # r = np.load(os.path.join(out_path, 'r_pre.npy'))
    print(t)
    print(r)

    trans_pred = interpret_transforms(t, r)
    with open(os.path.join(out_path, 'trans_pred.json'), 'w') as outfile:
        json.dump(trans_pred, outfile, indent=4)

    with open(os.path.join(data_path, 'trans.json'), 'r') as json_file:
        trans = json.load(json_file)
    trans_gt = trans['trans_info']
    if isinstance(trans_gt, dict):
        trans_gt = [trans_gt]

    pcd_pred = get_pred_point_cloud(out_path, K=num_movable)
    # pcd_pred = get_pred_point_cloud(out_path, K=num_movable, iters=20)
    pcd_gt = get_gt_point_clouds(os.path.join(data_path, 'gt/'), K=num_movable, reverse=reverse)

    metrics_axis = eval_axis_metrics(trans_pred, trans_gt, reverse=reverse, out_path=out_path)
    metrics_axis_stat = stat_axis_metrics(metrics_axis)
    metrics_cd = eval_geo_metrics(pcd_pred, pcd_gt)
    with open(os.path.join(out_path, 'metrics.json'), 'w') as outfile:
        json.dump(metrics_axis | metrics_axis_stat | metrics_cd, outfile, indent=4)

if __name__ == '__main__':
    # PARIS
    if True:
        # K = 1
        # st = 'output/paris/usb_st'
        # ed = 'output/paris/usb_ed'
        # data = 'data/dta/USB_100109'
        # out = 'output/paris/usb'
        # rev = False

        # K = 1
        # st = 'output/paris/blade_st'
        # ed = 'output/paris/blade_ed'
        # data = 'data/dta/blade_103706'
        # out = 'output/paris/blade'
        # rev = False

        # K = 1
        # st = 'output/paris/storage_st'
        # ed = 'output/paris/storage_ed'
        # data = 'data/dta/storage_45135'
        # out = 'output/paris/storage'
        # rev = True

        # K = 1
        # st = 'output/paris/rf_st'
        # ed = 'output/paris/rf_ed'
        # data = 'data/dta/real_fridge'
        # out = 'output/paris/rf'
        # rev = False

        # K = 1
        # st = 'output/paris/rs_st'
        # ed = 'output/paris/rs_ed'
        # data = 'data/dta/real_storage'
        # out = 'output/paris/rs'
        # rev = False

        pass
    # fi

    # DTA (2)
    if True:
        # K = 2
        # st = 'output/dta/storage_st'
        # ed = 'output/dta/storage_ed'
        # data = 'data/dta_multi/storage_47254'
        # out = 'output/dta/storage'
        # rev = True

        # st = 'output/dta/fridge_st'
        # ed = 'output/dta/fridge_ed'
        # data = 'data/dta_multi/fridge_10489'
        # out = 'output/dta/fridge'
        # rev = True
        pass
    # fi

    # Ours hard (2)
    if True:
        # K = 5
        # st = 'output/single/tbr5_st'
        # ed = 'output/single/tbr5_ed'
        # data = 'data/teeburu34610'
        # out = 'output/tbr5'
        # rev = False

        # K = 10
        # st = 'output/single/str_st'
        # ed = 'output/single/str_ed'
        # data = 'data/sutoreeji47585'
        # out = 'output/str'
        # rev = False
        # cd_thr = -10
        # dbs_eps = 0.004

        pass
    # fi

    # Ours dumped
    if True:
        # K = 5
        # st = 'output/single/ob5_st'
        # ed = 'output/single/ob5_ed'
        # data = 'data/oobun7201'
        # out = 'output/ob5'
        # rev = False

        # K = 7
        # st = 'output/single/nf_st'
        # ed = 'output/single/nf_ed'
        # data = 'data/naifu2'
        # out = 'output/nf'
        # rev = False
        # cd_thr = -6

        # K = 3
        # st = 'output/single/st3_st'
        # ed = 'output/single/st3_ed'
        # data = 'data/sutoreeji48063'
        # out = 'output/st3'
        # rev = False

        pass
    # if

    # ArtGS (5)
    if True:
        # K = 3
        # st = 'output/artgs/oven_st'
        # ed = 'output/artgs/oven_ed'
        # data = 'data/artgs/oven_101908'
        # out = 'output/artgs/oven'
        # rev = False

        # K = 3
        # st = 'output/artgs/tbl3_st'
        # ed = 'output/artgs/tbl3_ed'
        # data = 'data/artgs/table_25493'
        # out = 'output/artgs/tbl3'
        # rev = True

        # K = 3
        # st = 'output/artgs/sto3_st'
        # ed = 'output/artgs/sto3_ed'
        # data = 'data/artgs/storage_45503'
        # out = 'output/artgs/sto3'
        # rev = True

        # K = 4
        # st = 'output/artgs/tbl4_st'
        # ed = 'output/artgs/tbl4_ed'
        # data = 'data/artgs/table_31249'
        # out = 'output/artgs/tbl4'
        # rev = False

        # K = 6
        # st = 'output/artgs/sto6_st'
        # ed = 'output/artgs/sto6_ed'
        # data = 'data/artgs/storage_47648'
        # out = 'output/artgs/sto6'
        # rev = True

        pass
    # fi

    # Ours (6)
    if True:
        K = 4
        st = 'output/single/tbr4_st'
        ed = 'output/single/tbr4_ed'
        data = 'data/teeburu34178'
        out = 'output/tbr4'
        rev = False

        # K = 6
        # st = 'output/single/sut_st'
        # ed = 'output/single/sut_ed'
        # data = 'data/sutoreeji40417'
        # out = 'output/sut'
        # rev = False

        # K = 2
        # st = 'output/single/mado_st'
        # ed = 'output/single/mado_ed'
        # data = 'data/uindou103238'
        # out = 'output/mado'
        # rev = False

        # K = 4
        # st = 'output/single/tee_st'
        # ed = 'output/single/tee_ed'
        # data = 'data/teeburu23372'
        # out = 'output/tee'
        # rev = False

        # K = 4
        # st = 'output/single/sto4_st'
        # ed = 'output/single/sto4_ed'
        # data = 'data/sutoreeji45759'
        # out = 'output/sto4'
        # rev = False

        # K = 3
        # st = 'output/single/te3_st'
        # ed = 'output/single/te3_ed'
        # data = 'data/teeburu33116'
        # out = 'output/te3'
        # rev = False

        pass
    # fi

    # from_pgsr = True
    # st = st.replace('single', 'pgsr_wbce')
    # ed = ed.replace('single', 'pgsr_wbce')

    # _______________________________________________________

    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))
    # exit(0)

    # cluster_demo(out, data, K, thr=cd_thr)
    part_init_demo(out, st, ed, K)
    joint_init_demo(out, st, K)
    exit(0)

    get_gt_motion_params(data, reverse=rev)

    art_optim_demo(out, st, ed, data, num_movable=K)
    # K = merge_subparts(out, K)

    # refinement_demo(out, st, data, num_movable=K)
    eval_demo(out, data, num_movable=K, reverse=rev)
    vis_axes_pp_demo(out, st)

    pass
