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

from main_utils import train_single, get_gaussians, print_motion_params, get_gt_motion_params, plot_hist, \
    mk_output_dir, init_mpp, get_ppp_from_gmm, get_ppp_from_gmm_v2, get_gt_motion_params_mp, eval_init_gmm_params, \
    modify_scaling

from misc import mp_training_demo_v2, mp_mp_optim_demo

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

def mask_init_demo(out_path, st_path, ed_path, data_path, thr=None, sig_scale=1.0):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    gaussians_st = get_gaussians(st_path, from_chk=True)
    gaussians_ed = get_gaussians(ed_path, from_chk=True)

    csn, csn_is, csn_shifted = init_mpp(gaussians_st, gaussians_ed, thr=thr, sig_scale=sig_scale)
    mask = (csn > 0.5).detach().cpu().numpy()

    gaussians_m = copy.deepcopy(gaussians_st).cancel_grads()
    gaussians_m.get_opacity_raw[~mask] = -1e514
    gaussians_m.save_ply(os.path.join(out_path, 'point_cloud/iteration_2/point_cloud.ply'))
    gaussians_s = copy.deepcopy(gaussians_st).cancel_grads()
    gaussians_s.get_opacity_raw[mask] = -1e514
    gaussians_s.save_ply(os.path.join(out_path, 'point_cloud/iteration_3/point_cloud.ply'))
    plot_hist(csn, os.path.join(out_path, 'cdn.png'))
    plot_hist(csn_is, os.path.join(out_path, 'cdn-is.png'))
    plot_hist(csn_shifted, os.path.join(out_path, 'cdn-shifted.png'))
    np.save(os.path.join(out_path, 'mask_pre_pre.npy'), mask)
    np.save(os.path.join(out_path, 'prob_pre.npy'), csn_shifted.detach().cpu().numpy())
    return mask

def init_pp_demo(out_path: str, st_path: str, ed_path: str, data_path:str, num_movable: int):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    gaussians_st = get_gaussians(st_path, from_chk=True)
    gaussians_ed = get_gaussians(ed_path, from_chk=True)

    _, _, mpp = init_mpp(gaussians_st, gaussians_ed)
    mask_s = (mpp < 0.5)
    ppp = get_ppp_from_gmm_v2(train_pts=gaussians_st[~mask_s].get_xyz, test_pts=gaussians_st.get_xyz, num=num_movable)
    part_indices = torch.argmax(ppp, dim=1)

    for i in range(num_movable):
        gaussians_st[~mask_s & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{11 + i}/point_cloud.ply')
        )
    gaussians_st[mask_s].save_ply(os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
    plot_hist(mpp, os.path.join(out_path, 'mpp.png'))
    plot_hist(ppp[:, 0], os.path.join(out_path, 'ppp0.png'), bins=200)
    np.save(os.path.join(out_path, 'mpp_init.npy'), mpp.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'ppp_init.npy'), ppp.detach().cpu().numpy())
    gaussians_st[part_indices == 0].save_ply(os.path.join(out_path, 'point_cloud/iteration_9/point_cloud.ply'))

def init_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    gaussians_st = get_gaussians(st_path, from_chk=True)
    gaussians_ed = get_gaussians(ed_path, from_chk=True)

    _, _, mpp = init_mpp(gaussians_st, gaussians_ed)
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
    gaussians_st[mask_s].save_ply(os.path.join(out_path, 'point_cloud/iteration_10/point_cloud.ply'))
    gaussians_st[part_indices == 0].save_ply(os.path.join(out_path, 'point_cloud/iteration_9/point_cloud.ply'))
    plot_hist(mpp, os.path.join(out_path, 'mpp.png'))
    plot_hist(ppp[:, 0], os.path.join(out_path, 'ppp0.png'), bins=200)
    np.save(os.path.join(out_path, 'mpp_init.npy'), mpp.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'mu_init.npy'), mu.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'sigma_init.npy'), sigma.detach().cpu().numpy())

def am_optim_demo(out_path: str, st_path: str, ed_path: str, data_path, thr=0.85):
    torch.autograd.set_detect_anomaly(False)
    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_ed = get_gaussians(ed_path, from_chk=True).cancel_grads()
    prob_pre = torch.tensor(np.load(os.path.join(out_path, 'prob_pre.npy')), device='cuda')

    am = ArticulationModel(gaussians_st)
    am.dataset.eval = True
    am.dataset.source_path = os.path.join(os.path.realpath(data_path), 'end')
    am.dataset.model_path = out_path
    am.set_init_prob(prob_pre)
    t, r = am.train(gt_gaussians=gaussians_ed)

    prob = am.get_prob.detach().cpu().numpy()
    mask = (prob > thr)
    gaussians_m = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_m.get_opacity_raw[~mask] = -1e514
    gaussians_m.save_ply(os.path.join(out_path, 'point_cloud/iteration_8/point_cloud.ply'))
    gaussians_s = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_s.get_opacity_raw[mask] = -1e514
    gaussians_s.save_ply(os.path.join(out_path, 'point_cloud/iteration_9/point_cloud.ply'))
    np.save(os.path.join(out_path, 't_pre.npy'), t.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'r_pre.npy'), r.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'mask_pre.npy'), mask)

def mp_training_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int, thr=0.85):
    torch.autograd.set_detect_anomaly(False)
    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_ed = get_gaussians(ed_path, from_chk=True).cancel_grads()
    mpp_init = torch.tensor(np.load(os.path.join(out_path, 'mpp_init.npy')), device='cuda')
    ppp_init = torch.tensor(np.load(os.path.join(out_path, 'ppp_init.npy')), device='cuda')

    am = MPArtModel(gaussians_st, num_movable)
    am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'), model_path=out_path)
    am.set_init_probabilities(prob=mpp_init, ppp=ppp_init)
    t, r = am.train(gt_gaussians=gaussians_ed)

    mpp = am.get_prob.detach().cpu().numpy()
    ppp = am.get_ppp.detach().cpu().numpy()
    mask = (mpp > thr)
    part_indices = np.argmax(ppp, axis=1)
    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    for i in range(num_movable):
        gaussians_st[mask & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{21 + i}/point_cloud.ply')
        )
    gaussians_st[~mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_20/point_cloud.ply'))
    gaussians_st[part_indices == 0].save_ply(os.path.join(out_path, 'point_cloud/iteration_19/point_cloud.ply'))
    np.save(os.path.join(out_path, 't_pre.npy'), [tt.detach().cpu().numpy() for tt in t])
    np.save(os.path.join(out_path, 'r_pre.npy'), [rr.detach().cpu().numpy() for rr in r])
    np.save(os.path.join(out_path, 'mask_pre.npy'), mask)
    np.save(os.path.join(out_path, 'part_indices_pre'), part_indices)

def gmm_am_optim_demo(out_path: str, st_path: str, ed_path: str, data_path: str, num_movable: int, thr=0.85):
    # torch.autograd.set_detect_anomaly(False)
    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_ed = get_gaussians(ed_path, from_chk=True).cancel_grads()

    am = GMMArtModel(gaussians_st, num_movable)
    am.set_dataset(source_path=os.path.join(os.path.realpath(data_path), 'end'), model_path=out_path)
    am.set_init_params(out_path, scaling_modifier=10)
    t, r = am.train(gt_gaussians=gaussians_ed)

    ppp = am.get_ppp().detach().cpu().numpy()
    part_indices = np.argmax(ppp, axis=1)
    mpp = am.get_prob.detach().cpu().numpy()
    mask = (mpp > thr)

    gaussians_st = get_gaussians(st_path, from_chk=True).cancel_grads()
    gaussians_st[part_indices == 0].save_ply(os.path.join(out_path, 'point_cloud/iteration_19/point_cloud.ply'))
    for i in range(num_movable):
        gaussians_st[mask & (part_indices == i)].save_ply(
            os.path.join(out_path, f'point_cloud/iteration_{21 + i}/point_cloud.ply')
        )
    gaussians_st[~mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_20/point_cloud.ply'))
    np.save(os.path.join(out_path, 'r_pre.npy'), [rr.detach().cpu().numpy() for rr in r])
    np.save(os.path.join(out_path, 't_pre.npy'), [tt.detach().cpu().numpy() for tt in t])
    np.save(os.path.join(out_path, 'mask_pre.npy'), mask)
    np.save(os.path.join(out_path, 'part_indices_pre'), part_indices)

def joint_optim_demo(out_path: str, st_path: str, data_path: str):
    torch.autograd.set_detect_anomaly(False)
    r_pre = np.load(os.path.join(out_path, 'r_pre.npy'))
    t_pre = np.load(os.path.join(out_path, 't_pre.npy'))
    mask_pre = torch.tensor(np.load(os.path.join(out_path, 'mask_pre.npy')), device='cuda')

    gaussians_st = get_gaussians(st_path, from_chk=True)
    amj = ArticulationModelJoint(gaussians_st, data_path, out_path, mask_pre)
    amj.set_init_params(t_pre, r_pre)
    t, r = amj.train()

    gaussians_canonical = get_gaussians(out_path, from_chk=False, iters=amj.opt.iterations - 1)
    gaussians_canonical[amj.mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_5/point_cloud.ply'))
    gaussians_canonical[~amj.mask].save_ply(os.path.join(out_path, 'point_cloud/iteration_6/point_cloud.ply'))
    np.save(os.path.join(out_path, 'mask_final.npy'), amj.mask.cpu().numpy())
    np.save(os.path.join(out_path, 't_final.npy'), t.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'r_final.npy'), r.detach().cpu().numpy())

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

if __name__ == '__main__':
    # st = 'output/tmp/ust'
    # ed = 'output/tmp/ued'
    # data = 'data/dta/USB_100109'
    # out = 'output/tmp/'

    # st = 'output/tmp/bst'
    # ed = 'output/tmp/bed'
    # data = 'data/dta/blade_103706'
    # out = 'output/tmp/'

    # get_gt_motion_params(data)

    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))
    # mask_init_demo(out, st, ed, data, sig_scale=1)
    # am_optim_demo(out, st, ed, data)
    # joint_optim_demo(out, st, data)
    # exit(0)

    ### multi-part
    K = 2

    st = 'output/storage_st'
    ed = 'output/storage_ed'
    data = 'data/dta_multi/storage_47254'
    out = 'output/storage'

    # st = 'output/fridge_st'
    # ed = 'output/fridge_ed'
    # data = 'data/dta_multi/fridge_10489'
    # out = 'output/fridge'

    get_gt_motion_params_mp(data, reverse=True)

    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))
    # init_pp_demo(out, st, ed, data, num_movable=K)
    # mp_training_demo(out, st, ed, data, num_movable=K)
    # mp_joint_optimization_demo(out, st, data, num_movable=K)

    # init_demo(out, st, ed, data, num_movable=K)
    # gmm_am_optim_demo(out, st, ed, data, num_movable=K)
    mp_joint_optimization_demo(out, st, data, num_movable=K)

    pass
