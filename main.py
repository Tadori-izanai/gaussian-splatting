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
from utils.general_utils import build_rotation, quat_mult, weighted_l2_loss_v2, weighted_l2_loss_v1

def eval_img_loss(image, gt_image, opt) -> torch.Tensor:
    ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    return loss

def eval_rigid_loss(gaussians: GaussianModel) -> torch.Tensor:
    curr_rot = gaussians.get_rotation
    relative_rot = quat_mult(curr_rot, gaussians.prev_rotation_inv)
    rotation = build_rotation(relative_rot)

    prev_offset = gaussians.prev_xyz[gaussians.neighbor_indices] - gaussians.prev_xyz[:, None]
    curr_offset = gaussians.get_xyz[gaussians.neighbor_indices] - gaussians.get_xyz[:, None]
    curr_offset_in_prev_coord = (rotation.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
    return weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offset, gaussians.neighbor_weight)

def eval_rot_loss(gaussians: GaussianModel) -> torch.Tensor:
    curr_rot = gaussians.get_rotation
    relative_rot = quat_mult(curr_rot, gaussians.prev_rotation_inv)
    return weighted_l2_loss_v2(
        relative_rot[gaussians.neighbor_indices], relative_rot[:, None], gaussians.neighbor_weight
    )

def eval_iso_loss(gaussians: GaussianModel) -> torch.Tensor:
    curr_offset = gaussians.get_xyz[gaussians.neighbor_indices] - gaussians.get_xyz[:, None]
    curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
    return weighted_l2_loss_v1(curr_offset_mag, gaussians.neighbor_dist, gaussians.neighbor_weight)

def eval_losses(opt, image, gt_image, gaussians: GaussianModel):
    rigid_loss, rot_loss, iso_loss = None, None, None
    img_loss = eval_img_loss(image, gt_image, opt)
    loss = img_loss
    if opt.rigid_weight is not None:
        rigid_loss = eval_rigid_loss(gaussians)
        loss += opt.rigid_weight * rigid_loss
    if opt.rot_weight is not None:
        rot_loss = eval_rot_loss(gaussians)
        loss += opt.rot_weight * rot_loss
    if opt.iso_weight is not None:
        iso_loss = eval_iso_loss(gaussians)
        loss += opt.iso_weight * iso_loss
    return loss, img_loss, rigid_loss, rot_loss, iso_loss

def show_losses(iteration: int, img_loss, rigid_loss, rot_loss, iso_loss):
    if iteration in [10, 100, 1000, 5000, 10000, 20000, 30000]:
        loss_msg = f"\nimg {img_loss:.{7}f}"
        for loss, name in zip([rigid_loss, rot_loss, iso_loss], ['rigid', 'rot', 'iso']):
            if loss is not None:
                loss_msg += f"  {name} {loss:.{7}f}"
        print(loss_msg)

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
        gt_image = viewpoint_cam.original_image.cuda()

        loss, img_loss, rigid_loss, rot_loss, iso_loss = eval_losses(opt, image, gt_image, gaussians)
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
        show_losses(iteration, img_loss, rigid_loss, rot_loss, iso_loss)
    progress_bar.close()
    torch.save((gaussians.capture(), opt.iterations), os.path.join(scene.model_path, 'chkpnt.pth'))

def fit_demo(st_path=None, ed_path=None):
    dataset, pipes, opt = get_default_args()
    if not os.path.exists(ed_path):
        safe_state(False)
        torch.autograd.set_detect_anomaly(False)

        dataset.eval = True
        dataset.sh_degree = 0

        if not os.path.exists(st_path):
            print('Start checkpoint not found. Optimizing')
            gaussians = GaussianModel(dataset.sh_degree)
            dataset.source_path = os.path.realpath("data/USB100109/start")
            dataset.model_path = st_path
            train(dataset, opt, pipes, gaussians=gaussians, prev_iters=0)
            # checkpoint is saved to os.path.join(st_path, 'chkpnt.pth')

        print('End checkpoint not found. Optimizing')
        st_checkpoint = os.path.join(st_path, 'chkpnt.pth')
        model_params, prev_iters = torch.load(st_checkpoint)
        gaussians = GaussianModel(dataset.sh_degree).restore(model_params, opt)
        gaussians.initialize_neighbors(20)

        opt.densify_until_iter = 0
        opt.feature_lr = 0
        opt.opacity_lr = 0
        opt.scaling_lr = 0
        opt.rigid_weight = 1
        opt.rot_weight = 1
        opt.iso_weight = 100
        dataset.source_path = os.path.realpath("data/USB100109/end")
        dataset.model_path = ed_path
        train(dataset, opt, pipes, gaussians=gaussians, prev_iters=prev_iters)

    st_checkpoint = os.path.join(st_path, 'chkpnt.pth')
    ed_checkpoint = os.path.join(ed_path, 'chkpnt.pth')
    model_params, _ = torch.load(st_checkpoint)
    gaussians_st = GaussianModel(dataset.sh_degree).restore(model_params, opt)
    model_params, _ = torch.load(ed_checkpoint)
    gaussians_ed = GaussianModel(dataset.sh_degree).restore(model_params, opt)
    # todo

if __name__ == '__main__':
    st = 'output/fit_st'
    ed = 'output/fit_ed-1_1_100'
    fit_demo(st, ed)
    exit(0)

    safe_state(False)
    torch.autograd.set_detect_anomaly(False)

    dataset, pipes, opt = get_default_args()
    dataset.eval = True
    dataset.sh_degree = 0

    # trains "start"
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.source_path = os.path.realpath("data/USB100109/start")
    train(dataset, opt, pipes, gaussians=gaussians, prev_iters=0)

    gaussians.initialize_neighbors(20)

    # trains "end" from "start" with no densification
    opt.densify_until_iter = 0
    # and optimizes only positions and rotations
    opt.feature_lr = 0
    opt.opacity_lr = 0
    opt.scaling_lr = 0
    # adds physical priors
    opt.rigid_weight = 1
    opt.rot_weight = 1
    opt.iso_weight = 100
    # sets the dataset to "end" and trains
    dataset.source_path = os.path.realpath("data/USB100109/end")
    train(dataset, opt, pipes, gaussians=gaussians, prev_iters=opt.iterations)  # the same output model_path as above
