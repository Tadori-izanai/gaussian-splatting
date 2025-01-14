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
from utils.general_utils import otsu_with_peak_filtering, inverse_sigmoid
from scene.articulation_model import ArticulationModelBasic, ArticulationModelJoint
from scene.art_models import ArticulationModel

from main_utils import *

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

def am_seg(st_path: str, out_path: str, thresh=.85):
    prob = np.load(os.path.join(out_path, 'prob.npy'))

    gaussians_st = get_gaussians(st_path).cancel_grads()
    gaussians_st.get_opacity_raw[prob < thresh] = -1e514
    gaussians_st.save_ply(os.path.join(out_path, 'point_cloud/iteration_2/point_cloud.ply'))

    gaussians_st = get_gaussians(st_path).cancel_grads()
    gaussians_st.get_opacity_raw[prob > thresh] = -1e514
    gaussians_st.save_ply(os.path.join(out_path, 'point_cloud/iteration_3/point_cloud.ply'))

    # gaussians_st = GaussianModel(0).load_ply(os.path.join(st_path, 'point_cloud/iteration_30003/point_cloud.ply')).cancel_grads()
    # gaussians_st.get_opacity_raw[(prob > .6) | (prob < .4)] = -1e514
    # gaussians_st.save_ply(os.path.join(out_path, 'point_cloud/iteration_5/point_cloud.ply'))

    plt.figure()
    plt.hist(prob, bins=100)
    plt.savefig(os.path.join(out_path, 'disp-dist.png'))

def am_training_demo(st_path, out_path, gt_path, data_path, thresh=.85):
    os.makedirs(out_path, exist_ok=True)

    dataset, pipes, opt = get_default_args()
    model_params, _ = torch.load(os.path.join(st_path, 'chkpnt.pth'))
    gaussians_st = GaussianModel(0).restore(model_params, opt)
    gaussians_gt = GaussianModel(0).load_ply(os.path.join(gt_path, 'point_cloud/iteration_30000/point_cloud.ply'))

    am = ArticulationModel(gaussians_st)
    am.dataset.eval = True
    am.dataset.source_path = os.path.join(os.path.realpath(data_path), 'end')
    am.dataset.model_path = out_path
    t, r = am.train(gt_gaussians=gaussians_gt)

    prob = am.get_prob.detach().cpu().numpy()
    np.save(os.path.join(out_path, 'prob.npy'), prob)
    np.save(os.path.join(out_path, 't_pre.npy'), t.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'r_pre.npy'), r.detach().cpu().numpy())
    np.save(os.path.join(out_path, 'mask_pre.npy'), prob > thresh)

def am_training_with_gt_motion(st_path, out_path, gt_path, data_path, thresh=.85):
    gaussians_st = get_gaussians(st_path).cancel_grads()
    gaussians_gt = get_gaussians(gt_path).cancel_grads()

    am = ArticulationModel(gaussians_st)
    am.set_init_params(*get_gt_motion_params(data_path))
    am.dataset.eval = True
    am.dataset.source_path = os.path.join(os.path.realpath(data_path), 'end')
    am.dataset.model_path = out_path
    am.cancel_se3_grads()
    am.opt.cd_weight = None
    am.train(gt_gaussians=gaussians_gt)

    prob = am.get_prob.detach().cpu().numpy()
    np.save(os.path.join(out_path, 'prob.npy'), prob)
    np.save(os.path.join(out_path, 'mask.npy'), prob > thresh)

if __name__ == '__main__':
    st = 'output/usb_st'
    ed = 'output/usb_ed'
    data = 'data/USB100109'
    train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))

    # st = 'output/blade_st'
    # ed = 'output/blade_ed'
    # data = 'data/blade103706'
    # train_single_demo(st, os.path.join(data, 'start'))
    # train_single_demo(ed, os.path.join(data, 'end'))

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
