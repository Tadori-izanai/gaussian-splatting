import copy
import os
import torch
from random import randint
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
from utils.general_utils import otsu_with_peak_filtering, inverse_sigmoid, get_per_point_cd
from scene.articulation_model import ArticulationModelBasic, ArticulationModelJoint
from scene.art_models import ArticulationModel

from main_utils import train_single, get_gaussians, print_motion_params, get_gt_motion_params, plot_hist

def train_single_demo(path, data_path):
    dataset, pipes, opt = get_default_args()
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    dataset.eval = True
    dataset.sh_degree = 0
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.source_path = os.path.realpath(data_path)
    dataset.model_path = path
    # train_single(dataset, opt, pipes, gaussians, bce_weight=0.01)
    train_single(dataset, opt, pipes, gaussians, bce_weight=None)

def mask_init_demo(out_path, st_path, ed_path, data_path, thr=None):
    os.makedirs(out_path, exist_ok=True)
    dataset, pipes, opt = get_default_args()
    dataset.eval = True
    dataset.sh_degree = 0
    dataset.source_path = os.path.realpath(os.path.join(data_path, 'start'))
    dataset.model_path = out_path
    _ = prepare_output_and_logger(dataset)
    gaussians_st = get_gaussians(st_path, from_chk=True)
    gaussians_ed = get_gaussians(ed_path, from_chk=True)
    cds_st = get_per_point_cd(gaussians_st, gaussians_ed)
    cds_st_normalized = cds_st / torch.max(cds_st)

    eps = 1e-6
    csn_is = inverse_sigmoid(torch.clamp(cds_st_normalized, eps, 1-eps))
    if thr is None:
        thr = otsu_with_peak_filtering(csn_is.detach().cpu().numpy(), bias_factor=1.25)
        print(thr)
    csn_shifted = torch.sigmoid((inverse_sigmoid(cds_st_normalized) - thr))

    mask = (csn_shifted > 0.5).detach().cpu().numpy()

    gaussians_m = copy.deepcopy(gaussians_st).cancel_grads()
    gaussians_m.get_opacity_raw[~mask] = -1e514
    gaussians_m.save_ply(os.path.join(out_path, 'point_cloud/iteration_2/point_cloud.ply'))
    gaussians_s = copy.deepcopy(gaussians_st).cancel_grads()
    gaussians_s.get_opacity_raw[mask] = -1e514
    gaussians_s.save_ply(os.path.join(out_path, 'point_cloud/iteration_3/point_cloud.ply'))
    plot_hist(cds_st_normalized, os.path.join(out_path, 'cdn.png'))
    plot_hist(csn_is, os.path.join(out_path, 'cdn-is.png'))
    plot_hist(csn_shifted, os.path.join(out_path, 'cdn-shifted.png'))
    np.save(os.path.join(out_path, 'mask_pre_pre.npy'), mask)
    np.save(os.path.join(out_path, 'prob_pre.npy'), csn_shifted.detach().cpu().numpy())
    return mask

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

def joint_optim_demo(model_path: str, data_path='data/USB100109', pre_path=None):
    if pre_path is None:
        pre_path = model_path
    t = np.load(os.path.join(pre_path, 't_pre.npy'))
    r = np.load(os.path.join(pre_path, 'r_pre.npy'))
    mask = torch.tensor(np.load(os.path.join(pre_path, 'mask_pre.npy')), device='cuda')

    gaussians_st = get_gaussians(model_path, from_chk=True)
    am = ArticulationModelJoint(gaussians_st, data_path, model_path, mask)
    am.set_init_params(t, r)
    t, r = am.train()

    np.save(os.path.join(model_path, 'mask_final.npy'), am.mask.cpu().numpy())
    np.save(os.path.join(model_path, 't_final.npy'), t.detach().cpu().numpy())
    np.save(os.path.join(model_path, 'r_final.npy'), r.detach().cpu().numpy())
    gaussians_static = GaussianModel(0).load_ply(os.path.join(model_path, 'point_cloud/iteration_30000/point_cloud.ply')).cancel_grads()
    gaussians_static.get_opacity_raw[am.mask] = -1e514
    gaussians_static.save_ply(os.path.join(model_path, 'point_cloud/iteration_8/point_cloud.ply'))
    gaussians_dynamic = GaussianModel(0).load_ply(os.path.join(model_path, 'point_cloud/iteration_30000/point_cloud.ply')).cancel_grads()
    gaussians_dynamic.get_opacity_raw[~am.mask] = -1e514
    gaussians_dynamic.save_ply(os.path.join(model_path, 'point_cloud/iteration_9/point_cloud.ply'))

if __name__ == '__main__':
    # st = 'output/usb_st'
    # ed = 'output/usb_ed'
    # data = 'data/USB100109'
    # out = 'output/usb-art'

    st = 'output/blade_st'
    ed = 'output/blade_ed'
    data = 'data/blade103706'
    out = 'output/blade-art'

    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))
    # mask_init_demo(out, st, ed, data, thr=None)
    am_optim_demo(out, st, ed, data)
    #

    # st = 'output/st'
    # gt = 'output/ed'
    # out = 'output/trained_ed-v2'
    # am_training_demo(st, out, gt, data_path='data/USB100109')
    # am_seg(st, out, thresh=.85)
    # joint_optim_demo('output/st-art', data_path='data/USB100109', pre_path=out)

    # am_training_with_gt_motion(st, 'output/usb-gt_motion', gt, data_path='data/USB100109')
    # am_seg(st, 'output/usb-gt_motion')

    # st = 'output/blade_st'
    # gt = 'output/blade_ed'
    # out = 'output/blade_trained_v2'
    # am_training_demo(st, out, gt, data_path='data/blade103706')
    # am_seg(st, out, thresh=.85)
    # joint_optim_demo('output/blade_st-art', data_path='data/blade103706', pre_path=out)

    # am_training_with_gt_motion(st, 'output/blade-gt_motion', gt, data_path='data/blade103706')
    # am_seg(st, 'output/blade-gt_motion', thresh=.92)

    pass
