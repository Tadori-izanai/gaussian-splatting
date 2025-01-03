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

import matplotlib.pyplot as plt
import numpy as np
from train import prepare_output_and_logger
from arguments import get_default_args
from utils.loss_utils import eval_losses, show_losses
from utils.general_utils import otsu_with_peak_filtering, inverse_sigmoid
from scene.articulation_model import ArticulationModelLite, ArticulationModel0, ArticulationModel

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

def show_change(st_path, ed_path):
    gaussians_st, gaussians_ed, _ = get_two_gaussians(st_path, ed_path, from_chk=False)
    print(gaussians_st.get_scaling[:5] - gaussians_ed.get_scaling[:5])
    print(gaussians_st.get_opacity[:5] - gaussians_ed.get_opacity[:5])

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

def deform_demo(st_path, ed_path):
    gaussians_st, gaussians_ed, mask = get_two_gaussians(st_path, ed_path, from_chk=False)
    # am = ArticulationModelLite(gaussians_st)
    am = ArticulationModel0(gaussians_st)
    am.get_t = torch.tensor([-.344, 0, 0], dtype=torch.float, device='cuda')
    am.get_r = torch.tensor([[ 1, 1, 0], [0, 1, 0]], dtype=torch.float, device='cuda')
    gaussians_deformed = am.deform(mask)
    gaussians_deformed.save_ply(os.path.join(st_path, 'point_cloud/iteration_3/point_cloud.ply'))

def joint_optim_demo(st_path, ed_path, gt_path, data_path='data/USB100109/'):
    gaussians_st, gaussians_ed, mask = get_two_gaussians(st_path, ed_path, cancel_grad=True, from_chk=True)
    gaussians_gt = GaussianModel(0).load_ply(os.path.join(gt_path, 'point_cloud/iteration_30000/point_cloud.ply'))
    gaussians_gt.cancel_grads()

    am = ArticulationModelLite(gaussians_st)
    am.dataset.source_path = os.path.join(os.path.realpath(data_path), 'end')
    am.dataset.model_path = ed_path
    am.train(mask, gaussians_gt)
    # gaussians_deformed = am.deform(mask)
    # gaussians_deformed.save_ply(os.path.join(ed_path, 'point_cloud/iteration_3/point_cloud.ply'))

def am2_seg(st_path: str, out_path: str, thresh=.85):
    prob = np.load(os.path.join(out_path, 'prob.npy'))

    gaussians_st = GaussianModel(0).load_ply(os.path.join(st_path, 'point_cloud/iteration_30000/point_cloud.ply')).cancel_grads()
    gaussians_st.get_opacity_raw[prob < thresh] = -1e514
    gaussians_st.save_ply(os.path.join(out_path, 'point_cloud/iteration_2/point_cloud.ply'))

    gaussians_st = GaussianModel(0).load_ply(os.path.join(st_path, 'point_cloud/iteration_30000/point_cloud.ply')).cancel_grads()
    gaussians_st.get_opacity_raw[prob > thresh] = -1e514
    gaussians_st.save_ply(os.path.join(out_path, 'point_cloud/iteration_3/point_cloud.ply'))

    plt.figure()
    plt.hist(prob, bins=100)
    plt.savefig(os.path.join(out_path, 'disp-dist.png'))

def optim_demo(st_path, ed_path, gt_path, data_path):
    os.makedirs(ed_path, exist_ok = True)

    dataset, pipes, opt = get_default_args()
    model_params, _ = torch.load(os.path.join(st_path, 'chkpnt.pth'))
    gaussians_st = GaussianModel(0).restore(model_params, opt)
    gaussians_gt = GaussianModel(0).load_ply(os.path.join(gt_path, 'point_cloud/iteration_30000/point_cloud.ply'))

    # am = ArticulationModel0(gaussians_st)
    am = ArticulationModel(gaussians_st)
    am.dataset.eval = True
    am.dataset.source_path = os.path.join(os.path.realpath(data_path), 'end')
    am.dataset.model_path = ed_path
    am.train(gt_gaussians=gaussians_gt)
    np.save(os.path.join(ed_path, 'prob.npy'), am.get_prob.detach().cpu().numpy())
    am2_seg(st_path, ed_path)

if __name__ == '__main__':
    # st = 'output/st'
    # ed = 'output/ed-1_01_1-c20-u10k'
    # train_from_st(out_path=ed, st_path=st, gt_path='output/ed')
    # seg_demo(st, ed, thresh=None)
    # deform_demo(st, ed)
    # joint_optim_demo(st, ed, 'output/ed')

    # st = 'output/blade_st'
    # ed = 'output/blade_ed-iso1-c40-u10k'
    # train_from_st(out_path=ed, st_path=st, gt_path='output/blade_ed', data_path='data/blade103706')
    # seg_demo(st, ed, thresh=None)
    # joint_optim_demo(st, ed, 'output/blade_ed', data_path='data/blade103706')

    # st = 'output/st'
    # gt = 'output/ed'
    # out = 'output/trained_ed-v2'
    # optim_demo(st, out, gt, data_path='data/USB100109')
    # am2_seg(st, out, thresh=.9)

    st = 'output/blade_st'
    gt = 'output/blade_ed'
    out = 'output/blade_trained_v2'
    optim_demo(st, out, gt, data_path='data/blade103706')
    # am2_seg(st, out, thresh=.9)
