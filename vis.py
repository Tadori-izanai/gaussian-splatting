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

from os.path import join as pjoin

import json
import math
import numpy as np
from arguments import get_default_args
from main_utils import get_gaussians

from utils.general_utils import mat2quat, quat_mult

def deform(target: GaussianModel, source: GaussianModel, trans_info: dict, time):
    d = torch.tensor(trans_info['axis']['d'], dtype=target.get_xyz.dtype, device=target.get_xyz.device)
    o = torch.tensor(trans_info['axis']['o'], dtype=target.get_xyz.dtype, device=target.get_xyz.device)
    if trans_info['type'] == 'translate':
        t = d / torch.norm(d) * trans_info['translate'] * time
        target.get_xyz[:] = source.get_xyz[:] + t
    elif trans_info['type'] == 'rotate':
        theta = math.radians(trans_info['rotate']) * time
        K = torch.tensor([
            [0, -d[2], d[1]],
            [d[2], 0, -d[0]],
            [-d[1], d[0], 0]
        ], dtype=target.get_xyz.dtype, device=target.get_xyz.device)
        I = torch.eye(3, dtype=target.get_xyz.dtype, device=target.get_xyz. device)
        r = I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
        t = o - r @ o
        target.get_xyz[:] = torch.einsum('ij,nj->ni', r, source.get_xyz[:]) + t
        target.get_rotation_raw[:] = quat_mult(mat2quat(r), source.get_rotation[:])
    pass

def render_view(num_movable: int, model_path: str, data_path: str, view_idx: int, iteration: int=30, num_frames: int=120):
    with torch.no_grad():
        dataset, pipes, opt = get_default_args()
        dataset.eval = True
        dataset.sh_degree = 0
        dataset.source_path = os.path.realpath(pjoin(data_path, 'start'))
        dataset.model_path = model_path
        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        parts = [get_gaussians(model_path, from_chk=False, iters=iteration + k) for k in range(num_movable + 1)]
        deformed_parts = [get_gaussians(model_path, from_chk=False, iters=iteration + k) for k in range(1, num_movable + 1)]
        scene = Scene(dataset, parts[0], load_iteration=iteration, shuffle=False)

        with open(pjoin(model_path, 'trans_pred.json'), 'r') as json_file:
            trans = json.load(json_file)

        render_path = pjoin(model_path, f'train/view_{view_idx}')
        makedirs(render_path, exist_ok=True)
        view = scene.getTrainCameras()[view_idx]
        times = np.concat((
            np.linspace(0., 1., num_frames // 2), np.linspace(1., 0., num_frames // 2)
        ))
        for idx, time in enumerate(tqdm(times, desc="Rendering progress")):
            gaussians = parts[0]
            for k in range(num_movable):
                deform(deformed_parts[k], parts[k + 1], trans[k], time)
                gaussians += deformed_parts[k]

            render_pkg = render(view, gaussians, pipes, background)
            rendering = render_pkg["render"]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

if __name__ == '__main__':
    K = 4
    data = 'data/teeburu34178'
    out = 'output/tbr4'
    index = 11

    render_view(K, out, data, index)
    pass
