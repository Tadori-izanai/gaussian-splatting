#
# Created by lxl.
#

import os
import json
from os.path import join as pjoin
import numpy as np
from scene.dataset_readers import readCamerasFromTransforms
from scene.colmap_loader import proj_corr_to_world
from utils.general_utils import find_files_with_suffix

def load_json_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        info = json.load(json_file)
    return info

def get_cam(path: str):
    cam_infos = readCamerasFromTransforms(path, "transforms_train.json", False)
    trans = load_json_to_dict(pjoin(path, "transforms_train.json"))
    cam_info_dict = {}
    for frame, cam_info in zip(trans['frames'], cam_infos):
        cam_info_dict[frame['file_path']] = cam_info
    return cam_info_dict

def process_data(data_path: str):
    data_path = os.path.realpath(data_path)
    corr_path = pjoin(data_path, 'correspondence_loftr/no_filter')
    corr_files = find_files_with_suffix(corr_path, '.npz')

    src_cam_dict = get_cam(pjoin(data_path, 'start'))
    tgt_cam_dict = get_cam(pjoin(data_path, 'end'))

    for corr_file in corr_files:
        npz_file = pjoin(corr_path, corr_file)
        npz_data = np.load(npz_file, allow_pickle=True)['data']
        corr = npz_data[0]
        src_cam, tgt_cam = src_cam_dict[corr['src_path']], tgt_cam_dict[corr['tgt_path']]
        src_world_coords = proj_corr_to_world(src_cam, corr['src_corr'])
        tgt_world_coords = proj_corr_to_world(tgt_cam, corr['tgt_corr'])
        corr['src_world'], corr['tgt_world'] = src_world_coords, tgt_world_coords
        npz_data[0] = corr
        np.savez_compressed(npz_file, data=npz_data)
    print('done.')

if __name__ == '__main__':
    data = '../data/teeburu34610'
    process_data(data)
    pass
