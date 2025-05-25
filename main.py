import os
import torch

from scene import Scene, GaussianModel
from utils.general_utils import safe_state

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import json
import matplotlib.pyplot as plt
import numpy as np
from pytorch3d.loss import chamfer_distance
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from arguments import get_default_args
from utils.general_utils import eval_quad, inverse_sigmoid
from scene.multipart_models import MPArtModelJoint, GMMArtModel
from scene.deformable_model import DeformationModel, DMCanonical, DMGauFRe
from scene.dataset_readers import fetchPly, storePly

from main_utils import train_single, get_gaussians, print_motion_params, plot_hist, \
    mk_output_dir, init_mpp, get_ppp_from_gmm, get_ppp_from_gmm_v2, eval_init_gmm_params, \
    modify_scaling, get_vis_mask, value_to_rgb, estimate_se3, eval_mu_sigma
from metric_utils import get_gt_motion_params, interpret_transforms, eval_axis_metrics, \
    get_pred_point_cloud, get_gt_point_clouds, eval_geo_metrics

cd_thr = -5

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

    # clustering = DBSCAN(eps=0.01, min_samples=num_movable).fit(x)
    clustering = DBSCAN(eps=0.008, min_samples=num_movable).fit(x)

    labels = clustering.labels_
    pts = sorted(
        [(k, x[labels == k]) for k in np.unique_values(labels)],
        key=lambda item: len(item[1]), reverse=True
    )
    for k, pcd in pts[:num_movable]:
        storePly(os.path.join(ply_path, f'clusters/points3d_{k}.ply'), pcd, np.zeros_like(pcd))
    storePly(os.path.join(ply_path, f'points3d.ply'), x, np.zeros_like(x))

def init_demo_from_dbscan(out_path, st_path, ed_path, num_movable: int):
    gaussians_st = get_gaussians(st_path, from_chk=True)
    gaussians_ed = get_gaussians(ed_path, from_chk=True)
    cd, cd_is, mpp = init_mpp(gaussians_st, gaussians_ed, thr=-4.5)
    mask_m = (mpp > .5)
    gaussians_st[mask_m].save_ply(
        os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
    plot_hist(mpp, os.path.join(out_path, 'mpp.png'))
    np.save(os.path.join(out_path, 'mpp_init.npy'), mpp.detach().cpu().numpy())
    # return

    pts = []
    cluster_dir = os.path.join(out_path, 'clustering/clusters')
    for i in np.arange(20):
        ply_file = os.path.join(cluster_dir, f'points3d_{i}.ply')
        if not os.path.exists(ply_file):
            continue
        pts.append(np.asarray(fetchPly(ply_file).points))
        if len(pts) == num_movable:
            break
    assert len(pts) == num_movable

    mu = np.zeros((num_movable, 3))
    sigma = np.zeros((num_movable, 3, 3))
    for i in np.arange(num_movable):
        mu[i], sigma[i] = eval_mu_sigma(pts[i])
    np.save(os.path.join(out_path, 'mu_init.npy'), mu)
    np.save(os.path.join(out_path, 'sigma_init.npy'), sigma)

def gmm_am_optim_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int, thr=0.85):
    torch.autograd.set_detect_anomaly(False)
    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_ed = get_gaussians(ed_path, from_chk=True).cancel_grads()

    # am = GMMArtModel(gaussians_st, num_movable)
    am = GMMArtModel(gaussians_st, num_movable, new_scheme=False)
    am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'), model_path=out_path, thr=cd_thr)
    # am.set_init_params(out_path, scaling_modifier=10)
    am.set_init_params(out_path, scaling_modifier=1)
    am.save_ppp_vis(os.path.join(out_path, 'point_cloud/iteration_9/point_cloud.ply'))
    # return
    t, r = am.train(gt_gaussians=gaussians_ed)

    ppp = am.get_ppp().detach().cpu().numpy()
    part_indices = np.argmax(ppp, axis=1)
    mpp = am.get_prob.detach().cpu().numpy()
    mask = (mpp > thr)

    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    # gaussians_st[part_indices == 0].save_ply(os.path.join(out_path, 'point_cloud/iteration_19/point_cloud.ply'))
    am.save_ppp_vis(os.path.join(out_path, 'point_cloud/iteration_19/point_cloud.ply'))
    for i in range(num_movable):
        gaussians_st[mask & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{21 + i}/point_cloud.ply')
        )
    gaussians_st[~mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_20/point_cloud.ply'))
    np.save(os.path.join(out_path, 'r_pre.npy'), [rr.detach().cpu().numpy() for rr in r])
    np.save(os.path.join(out_path, 't_pre.npy'), [tt.detach().cpu().numpy() for tt in t])
    np.save(os.path.join(out_path, 'mask_pre.npy'), mask)
    np.save(os.path.join(out_path, 'part_indices_pre'), part_indices)

def mp_joint_optimization_demo(out_path: str, st_path: str, data_path: str, num_movable: int):
    torch.autograd.set_detect_anomaly(False)

    gaussians_st = get_gaussians(st_path, from_chk=True)
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
    metrics_cd = eval_geo_metrics(pcd_pred, pcd_gt)
    with open(os.path.join(out_path, 'metrics.json'), 'w') as outfile:
        json.dump(metrics_axis | metrics_cd, outfile, indent=4)

if __name__ == '__main__':
    ### paris and dta
    # K = 1
    # st = 'output/paris/usb_st'
    # ed = 'output/paris/usb_ed'
    # data = 'data/dta/USB_100109'
    # out = 'output/paris/usb'
    # rev = True

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

    ################## ArtGS ##################
    # K = 3
    # st = 'output/artgs/oven_st'
    # ed = 'output/artgs/oven_ed'
    # data = 'data/artgs/oven_101908'
    # out = 'output/artgs/oven'
    # rev = False

    # st = 'output/artgs/tbl3_st'
    # ed = 'output/artgs/tbl3_ed'
    # data = 'data/artgs/table_25493'
    # out = 'output/artgs/tbl3'
    # rev = True

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

    ################## Ours ##################
    K = 4
    st = 'output/tbr4_st'
    ed = 'output/tbr4_ed'
    data = 'data/teeburu34178'
    out = 'output/tbr4'
    rev = False

    # K = 5
    # st = 'output/tbr5_st'
    # ed = 'output/tbr5_ed'
    # data = 'data/teeburu34610'
    # out = 'output/tbr5'
    # rev = False

    # st = 'output/ob5_st'
    # ed = 'output/ob5_ed'
    # data = 'data/oobun7201'
    # out = 'output/ob5'
    # rev = False

    # K = 6
    # st = 'output/sut_st'
    # ed = 'output/sut_ed'
    # data = 'data/sutoreeji40417'
    # out = 'output/sut'
    # rev = False

    # K = 7
    # st = 'output/nf_st'
    # ed = 'output/nf_ed'
    # data = 'data/naifu2'
    # out = 'output/nf'
    # rev = False
    # cd_thr = -6

    get_gt_motion_params(data, reverse=rev)

    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))

    # cluster_demo(out, data, K, thr=cd_thr)
    # init_demo_from_dbscan(out, st, ed, K)

    gmm_am_optim_demo(out, st, ed, data, num_movable=K)
    # mp_joint_optimization_demo(out, st, data, num_movable=K)
    eval_demo(out, data, num_movable=K, reverse=rev)

    pass
