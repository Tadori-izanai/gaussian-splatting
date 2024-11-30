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
from arguments import GroupParams

def train(dataset, opt, pipe, gaussians=None, prev_iters=0):
    _ = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree)
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

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

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

    progress_bar.close()


def get_default_args() -> tuple:
    dataset_args = GroupParams()
    dataset_args.sh_degree = 3
    dataset_args.source_path = ""
    dataset_args.model_path = ""
    dataset_args.images = "images"
    dataset_args.resolution = -1
    dataset_args.white_background = False
    dataset_args.data_device = "cuda"
    dataset_args.eval = False

    pipes_args = GroupParams()
    pipes_args.convert_SHs_python = False
    pipes_args.compute_cov3D_python = False
    pipes_args.debug = False

    opt_args = GroupParams()
    opt_args.iterations = 30_000
    opt_args.position_lr_init = 0.00016
    opt_args.position_lr_final = 0.0000016
    opt_args.position_lr_delay_mult = 0.01
    opt_args.position_lr_max_steps = 30_000
    opt_args.feature_lr = 0.0025
    opt_args.opacity_lr = 0.05
    opt_args.scaling_lr = 0.005
    opt_args.rotation_lr = 0.001
    opt_args.percent_dense = 0.01
    opt_args.lambda_dssim = 0.2
    opt_args.densification_interval = 100
    opt_args.opacity_reset_interval = 3000
    opt_args.densify_from_iter = 500
    opt_args.densify_until_iter = 15_000
    opt_args.densify_grad_threshold = 0.0002
    opt_args.random_background = False

    return dataset_args, pipes_args, opt_args

if __name__ == '__main__':
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)
    dataset, pipes, opt = get_default_args()
    dataset.eval = True

    # trains "start"
    dataset.has_gaussians = False
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.source_path = os.path.realpath("data/USB100109/start")
    train(dataset, opt, pipes, gaussians=gaussians, prev_iters=0)

    # trains "end" from "start" with no densification
    opt.densify_until_iter = 0
    # and optimizes only positions and rotations
    opt.feature_lr = 0
    opt.opacity_lr = 0
    opt.scaling_lr = 0
    dataset.source_path = os.path.realpath("data/USB100109/end")
    train(dataset, opt, pipes, gaussians=gaussians, prev_iters=opt.iterations)  # the same output model_path as above
