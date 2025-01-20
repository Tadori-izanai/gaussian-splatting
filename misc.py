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

def train(dataset, opt, pipe, gaussians=None, gt_gaussians=None, prev_iters=0):
    _ = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians, is_new_gaussian=(prev_iters==0))
    if prev_iters == 0:
        gaussians.training_setup(opt)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(prev_iters + iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        loss, losses = eval_losses(opt, iteration, image, gt_image, gaussians, gt_gaussians)
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            # Save ply
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # copy or split
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                # opacity reset
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
        show_losses(iteration, losses)
    progress_bar.close()
    torch.save((gaussians.capture(), opt.iterations), os.path.join(scene.model_path, 'chkpnt.pth'))

def get_two_gaussians(st_path, ed_path, from_chk=True, cancel_grad=True) -> tuple[GaussianModel, GaussianModel, any]:
    gaussians_st, gaussians_ed = GaussianModel(0), GaussianModel(0)
    if from_chk:
        dataset, pipes, opt = get_default_args()
        model_params, _ = torch.load(os.path.join(st_path, 'chkpnt.pth'))
        gaussians_st.restore(model_params, opt)
        model_params, _ = torch.load(os.path.join(ed_path, 'chkpnt.pth'))
        gaussians_ed.restore(model_params, opt)
    else:
        gaussians_st.load_ply(os.path.join(st_path, 'point_cloud/iteration_30000/point_cloud.ply'))
        try:
            gaussians_ed.load_ply(os.path.join(ed_path, 'point_cloud/iteration_30000/point_cloud.ply'))
        except:
            gaussians_ed.load_ply(os.path.join(ed_path, 'point_cloud/iteration_60000/point_cloud.ply'))

    if cancel_grad:
        gaussians_st.cancel_grads()
        gaussians_ed.cancel_grads()

    displacements = (gaussians_ed.get_xyz - gaussians_st.get_xyz).norm(dim=1)
    normalized_displacements = displacements / torch.max(displacements)
    thresh = otsu_with_peak_filtering(normalized_displacements.detach().cpu().numpy())
    mask = normalized_displacements > thresh
    return gaussians_st, gaussians_ed, mask

def train_from_st(out_path, st_path, gt_path=None, data_path='data/USB100109/'):
    dataset, pipes, opt = get_default_args()
    model_params, prev_iters = torch.load(os.path.join(st_path, 'chkpnt.pth'))
    gaussians = GaussianModel(dataset.sh_degree).restore(model_params, opt)

    gaussians.initialize_neighbors(num_knn=20, lambda_omega=20)
    # gaussians.training_se3_setup(opt)

    gt_gaussians = None
    if gt_path is not None:
        model_params, _ = torch.load(os.path.join(gt_path, 'chkpnt.pth'))
        gt_gaussians = GaussianModel(dataset.sh_degree).restore(model_params, opt)

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)
    dataset.eval = True
    dataset.sh_degree = 0

    opt.densify_until_iter = 0
    # opt.rigid_weight = 1
    # opt.rot_weight = .1
    opt.iso_weight = 1
    opt.cd_weight = 40
    opt.cd_until_iter = 10_000
    dataset.source_path = os.path.realpath(os.path.join(data_path, 'end'))
    dataset.model_path = out_path
    train(dataset, opt, pipes, gaussians=gaussians, gt_gaussians=gt_gaussians, prev_iters=prev_iters)

def train_one(path, data_path):
    dataset, pipes, opt = get_default_args()
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    dataset.eval = True
    dataset.sh_degree = 0
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.source_path = os.path.realpath(data_path)
    dataset.model_path = path
    train(dataset, opt, pipes, gaussians=gaussians, prev_iters=0)

def seg_demo(st_path=None, ed_path=None, thresh=None):
    def get_rgb_form_displacement(disp: torch.Tensor):
        disp = torch.log(1 + 100 * disp + 1e-20) / torch.log(torch.as_tensor(1 + 100))
        r = torch.clamp(disp, 0., 1.)
        g = torch.clamp(1. - disp, 0., 1.)
        b = torch.clamp(.5 * disp, 0., 1.)
        rgb_tensor = torch.stack((r, g, b), dim=-1)
        return rgb_tensor

    gaussians_st, gaussians_ed, _ = get_two_gaussians(st_path, ed_path, from_chk=False)
    displacements = (gaussians_ed.get_xyz - gaussians_st.get_xyz).norm(dim=1)
    normalized_displacements = displacements / torch.max(displacements)

    if thresh is None:
        thresh = otsu_with_peak_filtering(normalized_displacements.detach().cpu().numpy())
        print('threshold:', thresh)

    print(normalized_displacements.shape)
    print(torch.count_nonzero(torch.Tensor(normalized_displacements < thresh)))
    plt.figure()
    plt.hist(normalized_displacements.detach().cpu().numpy(), bins=1000)
    plt.savefig(os.path.join(ed_path, 'disp-dist.png'))
    plt.savefig(os.path.join(st_path, 'disp-dist.png'))

    # rgb = get_rgb_form_displacement(normalized_displacements)
    # gaussians_st.set_colors(rgb)
    # gaussians_st.save_ply(os.path.join(st_path, 'point_cloud/iteration_1/point_cloud.ply'))
    # gaussians_ed.set_colors(rgb)
    # gaussians_ed.save_ply(os.path.join(ed_path, 'point_cloud/iteration_1/point_cloud.ply'))

    gaussians_st.hide_static(normalized_displacements, thresh)
    gaussians_st.save_ply(os.path.join(st_path, 'point_cloud/iteration_2/point_cloud.ply'))
    gaussians_ed.hide_static(normalized_displacements, thresh)
    gaussians_ed.save_ply(os.path.join(ed_path, 'point_cloud/iteration_2/point_cloud.ply'))

def motion_param_optim_demo(st_path, ed_path, gt_path, data_path='data/USB100109/'):
    gaussians_st, gaussians_ed, mask = get_two_gaussians(st_path, ed_path, cancel_grad=True, from_chk=True)
    gaussians_gt = GaussianModel(0).load_ply(os.path.join(gt_path, 'point_cloud/iteration_30000/point_cloud.ply'))
    gaussians_gt.cancel_grads()

    am = ArticulationModelBasic(gaussians_st, mask)
    am.dataset.source_path = os.path.join(os.path.realpath(data_path), 'end')
    am.dataset.model_path = ed_path
    t, r = am.train(gaussians_gt)

    gaussians_deformed = am.deform()
    gaussians_deformed.save_ply(os.path.join(ed_path, 'point_cloud/iteration_3/point_cloud.ply'))
    np.save(os.path.join(st_path, 't_pre.npy'), t.detach().cpu().numpy())
    np.save(os.path.join(st_path, 'r_pre.npy'), r.detach().cpu().numpy())
    np.save(os.path.join(st_path, 'mask_pre.npy'), mask.cpu().numpy())

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

def final_joint_optim_demo(model_path: str, data_path='data/USB100109', pre_path=None):
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
