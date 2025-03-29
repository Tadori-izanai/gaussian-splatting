import copy
import os
import torch
from random import randint

from torch.nn.attention.bias import causal_upper_left

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import json
import matplotlib.pyplot as plt
import numpy as np
from train import prepare_output_and_logger
from arguments import get_default_args
from utils.loss_utils import eval_losses, show_losses
from utils.general_utils import eval_quad
from scene.articulation_model import ArticulationModelBasic, ArticulationModelJoint
from scene.art_models import ArticulationModel
from scene.multipart_models import MPArtModel, MPArtModelJoint, GMMArtModel
from scene.multipart_misc import  OptimOMP, MPArtModelII

from main_utils import train_single, get_gaussians, print_motion_params, plot_hist, \
    mk_output_dir, init_mpp, get_ppp_from_gmm, get_ppp_from_gmm_v2, eval_init_gmm_params, \
    modify_scaling, get_vis_mask
from metric_utils import get_gt_motion_params, interpret_transforms, eval_axis_metrics, \
    get_pred_point_cloud, get_gt_point_clouds, eval_geo_metrics

from misc import mp_training_demo_v2, mp_mp_optim_demo
from render import render_depth_for_pcd

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

def init_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    gaussians_st = get_gaussians(st_path, from_chk=True)
    gaussians_ed = get_gaussians(ed_path, from_chk=True)

    # cd, cd_is, mpp = init_mpp(gaussians_st, gaussians_ed)
    cd, cd_is, mpp = init_mpp(gaussians_st, gaussians_ed, thr=-4)
    mask_s = (mpp < .5)
    mu, sigma = eval_init_gmm_params(train_pts=gaussians_st[~mask_s].get_xyz, num=num_movable)

    sigma_modified = modify_scaling(sigma, scaling_modifier=10)
    quad = eval_quad(gaussians_st.get_xyz.unsqueeze(1) - mu, torch.linalg.inv(sigma_modified))
    ppp = torch.exp(-quad)
    ppp /= ppp.sum(dim=1, keepdim=True)
    part_indices = torch.argmax(ppp, dim=1)

    for i in range(num_movable):
        gaussians_st[~mask_s & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{11 + i}/point_cloud.ply')
        )
    # gaussians_st[~mask_s].save_ply(os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
    gaussians_m = gaussians_st[~mask_s]
    gaussians_m[get_vis_mask(gaussians_m, os.path.join(data_path, 'end'))].save_ply(
        os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))

    plot_hist(cd, os.path.join(out_path, 'cd.png'))
    plot_hist(cd_is, os.path.join(out_path, 'cd_is.png'))
    plot_hist(mpp, os.path.join(out_path, 'mpp.png'))
    plot_hist(ppp[:, 0], os.path.join(out_path, 'ppp0.png'), bins=200)
    np.save(os.path.join(out_path, 'mpp_init.npy'), mpp.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'mu_init.npy'), mu.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'sigma_init.npy'), sigma.detach().cpu().numpy())

def gmm_am_optim_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int, thr=0.85):
    torch.autograd.set_detect_anomaly(False)
    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_ed = get_gaussians(ed_path, from_chk=True).cancel_grads()

    am = GMMArtModel(gaussians_st, num_movable)
    am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'), model_path=out_path)
    # am.set_init_params(out_path, scaling_modifier=10)
    am.set_init_params(out_path, scaling_modifier=1)
    am.save_ppp_vis(os.path.join(out_path, 'point_cloud/iteration_9/point_cloud.ply'))
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
    pcd_gt = get_gt_point_clouds(os.path.join(data_path, 'gt/'), K=num_movable, reverse=reverse)

    metrics_axis = eval_axis_metrics(trans_pred, trans_gt)
    metrics_cd = eval_geo_metrics(pcd_pred, pcd_gt)
    with open(os.path.join(out_path, 'metrics.json'), 'w') as outfile:
        json.dump(metrics_axis | metrics_cd, outfile, indent=4)

if __name__ == '__main__':
    ### paris and dta
    # K = 2
    # st = 'output/storage_st'
    # ed = 'output/storage_ed'
    # data = 'data/dta_multi/storage_47254'
    # out = 'output/storage'
    # rev = True

    # st = 'output/fridge_st'
    # ed = 'output/fridge_ed'
    # data = 'data/dta_multi/fridge_10489'
    # out = 'output/fridge'
    # rev = True

    # K = 1
    # st = 'output/usb_st'
    # ed = 'output/usb_ed'
    # data = 'data/dta/USB_100109'
    # out = 'output/usb'
    # rev = False

    # st = 'output/blade_st'
    # ed = 'output/blade_ed'
    # data = 'data/dta/blade_103706'
    # out = 'output/blade'
    # rev = False

    ### ArtGS
    K = 3
    # st = 'output/oven_st'
    # ed = 'output/oven_ed'
    # data = 'data/artgs/oven_101908'
    # out = 'output/oven'
    # rev = False

    st = 'output/tbl3_st'
    ed = 'output/tbl3_ed'
    data = 'data/artgs/table_25493'
    out = 'output/tbl3'
    rev = True

    # K = 6
    # st = 'output/sto6_st'
    # ed = 'output/sto6_ed'
    # data = 'data/artgs/storage_47648'
    # out = 'output/sto6'
    # rev = True

    ## outs
    # K = 4
    # st = 'output/tbr4_st'
    # ed = 'output/tbr4_ed'
    # data = 'data/teeburu34178'
    # out = 'output/tbr4'
    # rev = False

    get_gt_motion_params(data, reverse=rev)

    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))
    # init_demo(out, st, ed, data, num_movable=K)
    # gmm_am_optim_demo(out, st, ed, data, num_movable=K)
    # mp_joint_optimization_demo(out, st, data, num_movable=K)
    # eval_demo(out, data, num_movable=K, reverse=rev)

    pass
