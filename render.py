#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, get_source_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image

import json
from pathlib import Path
import numpy as np
from utils.camera_utils import create_transforms_on_sphere
from utils.graphics_utils import fov2focal, focal2fov
from utils.camera_utils import cameraList_from_camInfos
from scene.dataset_readers import CameraInfo
from arguments import get_default_args
from main_utils import get_gaussians

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, 'ours_{}'.format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        depth *= 1000
        depth = depth.to(torch.uint32).squeeze(0).cpu().numpy()
        Image.fromarray(depth, mode="I").save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

def render_depth_for_pcd(out_path: str, iteration: int):
    depth_dir = os.path.join(out_path, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    data_path = get_source_path(os.path.join(out_path, 'cfg_args'))

    transforms_depth = os.path.join(depth_dir, 'transforms.json')
    with open(os.path.join(data_path, 'transforms_train.json')) as json_file:
        contents = json.load(json_file)
        angle_x = contents["camera_angle_x"]
        frame = contents["frames"][0]
        c2w = np.array(frame["transform_matrix"])
        w2c = np.linalg.inv(c2w)
        radius = float(np.linalg.norm(w2c[:3, 3]))
        create_transforms_on_sphere(transforms_depth, radius, angle_x)

        cam_name = os.path.join(data_path, frame["file_path"] + '.png')
        image_path = os.path.join(data_path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        images_folder, image_basename = os.path.split(image_path)
        image_d_path = os.path.join(images_folder + '_d', image_basename).__str__()
        if os.path.exists(image_d_path):
            image_d = Image.open(image_d_path)
            image_d = np.array(image_d) / 1e3
        else:
            image_d_path = None
            image_d = None
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

    cam_infos = []
    with open(transforms_depth) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_d_path=image_d_path, image_d=image_d,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    dataset, pipeline, _ = get_default_args()
    dataset.eval = True
    dataset.sh_degree = 0
    dataset.source_path = os.path.realpath(data_path)
    dataset.model_path = out_path
    views = cameraList_from_camInfos(cam_infos, 1.0, dataset)

    depth_path = os.path.join(depth_dir, 'ours_{}'.format(iteration))
    gaussians = get_gaussians(out_path, from_chk=False, iters=iteration)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    makedirs(depth_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        depth = render_pkg["depth"]
        depth *= 1000
        depth = depth.to(torch.uint32).squeeze(0).cpu().numpy()
        Image.fromarray(depth, mode="I").save(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        cam_infos[idx] = cam_infos[idx]._replace(image_d=depth / 1e3)
        cam_infos[idx] = cam_infos[idx]._replace(image=None)

    return cam_infos
