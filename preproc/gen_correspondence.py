#
# Created by lxl.
#

import os
import sys
from os.path import join as pjoin
base_dir = os.path.dirname(__file__)
sys.path.insert(0, base_dir)
sys.path.insert(0, pjoin(base_dir, '..'))

import numpy as np
import argparse
import json
import cv2
from PIL import Image
from tqdm import tqdm
from loftr_wrapper import LoftrRunner
from corr2world import process_data as corr2world

def load_json_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        info = json.load(json_file)
    return info

def c2w_to_t(c2w: list) -> np.array:
    c2w = np.array(c2w)
    # c2w[:3, 1:3] *= -1
    # w2c = np.linalg.inv(c2w)
    # return w2c[:3, 3]
    return c2w[:3, 3]

def read_img_dict(path, frame, ext='.png'):
    img_file = pjoin(path, frame['file_path']) + ext
    img = np.array(Image.open(img_file))
    return {'rgb': img[..., :3], 'mask': img[..., -1]}

def draw_corr(rgbA, rgbB, corrA, corrB, output_name):
    vis = np.concatenate([rgbA, rgbB], axis=1)
    radius = 2
    for i in range(len(corrA)):
        if i % 20 != 0:
            continue
        uvA = corrA[i]
        uvB = corrB[i].copy()
        uvB[0] += rgbA.shape[1]
        color = tuple(np.random.randint(0, 255, size=(3)).tolist())
        vis = cv2.circle(vis, uvA, radius=radius, color=color, thickness=1)
        vis = cv2.circle(vis, uvB, radius=radius, color=color, thickness=1)
        vis = cv2.line(vis, uvA, uvB, color=color, thickness=1, lineType=cv2.LINE_AA)
    Image.fromarray(vis).save(f'{output_name}.png')

def compute_correspondence(model: LoftrRunner, src_dict, tgt_dict,
                           visualize=False, vis_path='test_loftr', filter_level_list=None):
    if filter_level_list is None:
        filter_level_list = ['no_filter']

    tgt_mask = tgt_dict['mask']
    tgt_rgb = tgt_dict['rgb']
    src_mask = src_dict['mask']
    src_rgb = src_dict['rgb']
    img_h, img_w = src_rgb.shape[:2]
    cur_corres = model.predict(src_rgb[np.newaxis], tgt_rgb[np.newaxis])[0]

    def get_valid_mask(mask, coords):
        valid = np.logical_and(np.logical_and(coords[..., 0] >= 0, coords[..., 0] < img_w),
                               np.logical_and(coords[..., 1] >= 0, coords[..., 1] < img_h))
        valid = np.logical_and(valid, mask[coords[..., 1], coords[..., 0]])
        return valid

    filtered_corr = {key: [] for key in filter_level_list}

    src_coords = cur_corres[:, :2].round().astype(int)
    tgt_coords = cur_corres[:, 2:4].round().astype(int)
    valid_mask = np.logical_and(get_valid_mask(src_mask, src_coords),
                                get_valid_mask(tgt_mask, tgt_coords))
    loftr_total = len(valid_mask)
    valid_total = sum(valid_mask)
    src_coords = src_coords[np.where(valid_mask)[0]]
    tgt_coords = tgt_coords[np.where(valid_mask)[0]]

    if 'no_filter' in filter_level_list:
        filtered_corr['no_filter'].append((src_coords, tgt_coords))
    if visualize:
        draw_corr(src_rgb, tgt_rgb, src_coords, tgt_coords, pjoin(vis_path, f'0_1_no_filter_{valid_total}_of_{loftr_total}'))
    return filtered_corr

def run_source_img(model, folder, src_frame,
                   filter_level_list=None, visualize=False, vis_path='test_loftr'):
    if filter_level_list is None:
        filter_level_list = ['no_filter']

    t_src = c2w_to_t(src_frame["transform_matrix"])

    tgt_frames = load_json_to_dict(pjoin(folder, 'end/transforms_train.json'))['frames']
    min_dist = np.inf
    tgt_frame = None
    for frame in tgt_frames:
        t_tgt = c2w_to_t(frame["transform_matrix"])
        dist =  np.linalg.norm(t_tgt - t_src)
        if dist < min_dist:
            min_dist = dist
            tgt_frame = frame

    src_dict = read_img_dict(os.path.join(folder, 'start'), src_frame)
    tgt_dict = read_img_dict(os.path.join(folder, 'end'), tgt_frame)

    all_corr = compute_correspondence(model, src_dict, tgt_dict, visualize, vis_path, filter_level_list)

    if visualize:
        src_coords, tgt_coords = all_corr['no_filter'][0]
        src_name = os.path.basename(src_frame['file_path'])
        tgt_name = os.path.basename(tgt_frame['file_path'])
        draw_corr(src_dict['rgb'], tgt_dict['rgb'], src_coords, tgt_coords,
                  pjoin(vis_path, f'no_filter_{src_name}_{tgt_name}'))

    return {
        filter_level: [
            # {src_frame['file_path']: tgt_corr[0], tgt_frame['file_path']: tgt_corr[1]} for tgt_corr in corr
            {'src_corr': tgt_corr[0], 'tgt_corr': tgt_corr[1],
             'src_path': src_frame['file_path'], 'tgt_path': tgt_frame['file_path']} for tgt_corr in corr
        ] for filter_level, corr in all_corr.items()
    }

def run_folder(model, folder):
    output_path = pjoin(folder, 'correspondence_loftr')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs('test_loftr', exist_ok=True)

    all_frames = load_json_to_dict(pjoin(folder, 'start/transforms_train.json'))['frames']
    src_names = [os.path.basename(frame['file_path']) for frame in all_frames]

    pbar = tqdm(src_names)
    for src_name, frame in zip(pbar, all_frames):
        pbar.set_description(src_name)
        results = run_source_img(model, folder, frame, visualize=True)
        for filter_level in results:
            os.makedirs(pjoin(output_path, filter_level), exist_ok=True)
            np.savez_compressed(
                pjoin(output_path, filter_level, f'src_{src_name}_tgt_all.npz'), data=results[filter_level]
            )

if __name__ == '__main__':
    data_path = '../data/naifu2'
    data_path = '../data/oobun7201'
    data_path = '../data/sutoreeji40417'
    data_path = '../data/teeburu34178'
    # data_path = '../data/teeburu34610'

    # data_path = '../data/artgs/oven_101908'
    # data_path = '../data/artgs/storage_45503'
    # data_path = '../data/artgs/storage_47648'
    # data_path = '../data/artgs/table_25493'
    # data_path = '../data/artgs/table_31249'

    loftr = LoftrRunner()
    run_folder(loftr, data_path)
    corr2world(data_path)
    pass
