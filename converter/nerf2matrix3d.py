import os
import json
import math
import shutil

from os.path import join as pjoin

RES = 800

def load_json_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        info = json.load(json_file)
    return info

def save_list_to_txt(lst: list, filename: str):
    with open(filename, 'w') as file:
        for f in lst:
            file.write(str(f) + '\n')

def process_state(source_path: str, target_path: str):
    transforms = load_json_to_dict(pjoin(source_path, 'transforms_train.json'))
    
    horizontal_fov = transforms['camera_angle_x']
    f_x = RES / (2 * math.tan(horizontal_fov / 2))
    cam_k = [f_x, f_x, RES / 2, RES / 2]

    n = len(transforms['frames'])
    p = math.ceil(n / 8)
    for i in range(p):
        os.makedirs(pjoin(target_path, str(i)), exist_ok=True)

    def process_frame(idx, frame):
        bucket = idx // 8
        bucket_path = pjoin(target_path, str(bucket))

        basename = os.path.basename(frame['file_path'])
        imagename = basename + '.png'

        ext = []
        for row in frame['transform_matrix']:
            ext += row

        save_list_to_txt(cam_k, pjoin(bucket_path, basename + '.txt'))
        save_list_to_txt(ext, pjoin(bucket_path, basename + '_ext.txt'))
        shutil.copy(pjoin(source_path, frame['file_path'] + '.png'), pjoin(bucket_path, imagename))

    for idx, frame in enumerate(transforms['frames']):
        process_frame(idx, frame)
    for idx in range(n, 8 * p):
        process_frame(idx, transforms['frames'][idx % n])

def convert_to_matrix3d(data_path: str):
    mat_path = pjoin(data_path, 'matrix3d')
    os.makedirs(mat_path, exist_ok=True)
    
    for state in ['start', 'end']:
        nerf_path = pjoin(data_path, state)
        out_path = pjoin(mat_path, state)
        os.makedirs(out_path, exist_ok=True)
        process_state(nerf_path, out_path)

if __name__ == '__main__':
    data = 'data/sutoreeji40417'
    data = 'data/teeburu34178'

    convert_to_matrix3d(data)

    pass
