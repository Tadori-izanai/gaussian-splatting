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

from train import prepare_output_and_logger
from arguments import get_default_args
from utils.loss_utils import eval_losses, show_losses
from utils.general_utils import otsu_with_peak_filtering
import matplotlib.pyplot as plt

def get_rgb_form_displacement(disp: torch.Tensor):
    disp = torch.log(1 + 100 * disp + 1e-20) / torch.log(torch.as_tensor(1 + 100))
    r = torch.clamp(disp, 0., 1.)
    g = torch.clamp(1. - disp, 0., 1.)
    b = torch.clamp(.5 * disp, 0., 1.)
    rgb_tensor = torch.stack((r, g, b), dim=-1)
    return rgb_tensor

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

def get_two_gaussians(st_path=None, ed_path=None) -> tuple[GaussianModel, GaussianModel]:
    dataset, pipes, opt = get_default_args()
    st_checkpoint = os.path.join(st_path, 'chkpnt.pth')
    ed_checkpoint = os.path.join(ed_path, 'chkpnt.pth')
    model_params, _ = torch.load(st_checkpoint)
    gaussians_st = GaussianModel(dataset.sh_degree).restore(model_params, opt)
    model_params, _ = torch.load(ed_checkpoint)
    gaussians_ed = GaussianModel(dataset.sh_degree).restore(model_params, opt)
    return gaussians_st, gaussians_ed

def seg_demo(st_path=None, ed_path=None, thresh=None):
    gaussians_st, gaussians_ed = get_two_gaussians(st_path, ed_path)
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

    gt_gaussians = None
    if gt_path is not None:
        model_params, _ = torch.load(os.path.join(gt_path, 'chkpnt.pth'))
        gt_gaussians = GaussianModel(dataset.sh_degree).restore(model_params, opt)

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)
    dataset.eval = True
    dataset.sh_degree = 0

    opt.densify_until_iter = 0
    opt.feature_lr = 0
    opt.opacity_lr = 0
    opt.scaling_lr = 0
    # opt.rigid_weight = 1
    # opt.rot_weight = .1
    opt.iso_weight = 10
    opt.cd_weight = 0.4
    opt.cd_until_iter = 10_000
    dataset.source_path = os.path.realpath(os.path.join(data_path, 'end'))
    dataset.model_path = out_path
    train(dataset, opt, pipes, gaussians=gaussians, gt_gaussians=gt_gaussians, prev_iters=prev_iters)

if __name__ == '__main__':
    st = 'output/st'
    ed = 'output/ed-1_01_1-c20-u10k'
    # train_from_st(out_path=ed, st_path=st, gt_path='output/ed')
    seg_demo(st, ed, thresh=None)

    # train_one('output/blade_st', 'data/blade103706/start')
    # train_one('output/blade_ed', 'data/blade103706/end')
    # train_from_st(
    #     out_path='output/blade_ed-iso1-c40-u10k', st_path='output/blade_st', gt_path='output/blade_ed',
    #     data_path='data/blade103706')
    # seg_demo(st_path='output/blade_st', ed_path='output/blade_ed-iso1-c40-u10k', thresh=None)
