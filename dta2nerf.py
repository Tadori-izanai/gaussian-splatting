import os
import json
import yaml
import math
import numpy as np
from PIL import Image, ImageFilter

def load_yaml_to_dict(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        ret = yaml.safe_load(f)
    return ret

def save_dict_to_json(data: dict, file_path: str) -> None:
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def get_img(img_path: str) -> np.ndarray:
    img = Image.open(img_path)
    return np.array(img)

def rgb_mask_to_rgba(img_rgb: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    hh, ww = img_mask.shape
    img_rgba = np.zeros((hh, ww, 4), dtype=np.uint8)
    img_rgba[:, :, :3] = img_rgb
    img_rgba[:, :, 3] = img_mask
    return img_rgba

def gen_rgba_images(path: str) -> None:
    rgb_path = os.path.join(path, 'color_segmented')
    mask_path = os.path.join(path, 'mask')
    output_path = os.path.join(path, 'color_rgba')
    os.makedirs(output_path, exist_ok=True)

    for img_file in os.listdir(rgb_path):
        if not img_file.endswith(".png"):
            continue
        img_rgb = get_img(os.path.join(rgb_path, img_file))
        img_mask = get_img(os.path.join(mask_path, img_file))
        img_rgba = rgb_mask_to_rgba(img_rgb, img_mask)
        image = Image.fromarray(img_rgba, mode="RGBA")
        image.verify()
        image.save(os.path.join(output_path, img_file))
        image.close()
        print('done with image:', img_file)

def convert_to_nerf(path: str, res_x: int, is_start_zero: bool) -> None:
    def gen_empty_dict():
        return {'camera_angle_x': horizontal_fov, 'frames': []}

    start_path = os.path.join(path, 'start')
    end_path = os.path.join(path, 'end')
    train_st_path = os.path.join(start_path, 'train')
    test_st_path = os.path.join(start_path, 'test')
    train_ed_path = os.path.join(end_path, 'train')
    test_ed_path = os.path.join(end_path, 'test')
    for out_path in [train_st_path, test_st_path, train_ed_path, test_ed_path]:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(out_path + '_d', exist_ok=True)

    cam_k = np.loadtxt(os.path.join(path, 'cam_K.txt'))
    f_x = cam_k[0][0]
    horizontal_fov = 2 * math.atan(res_x / (2 * f_x))

    transforms_train_st = gen_empty_dict()
    transforms_train_ed = gen_empty_dict()
    transforms_test = gen_empty_dict()
    cameras = load_yaml_to_dict(os.path.join(path, 'init_keyframes.yml'))
    for frame_img_name_no_ext, transform_info in cameras.items():
        is_zero = (transform_info['time'] == 0)
        is_end = (is_start_zero ^ is_zero)
        img_name_no_ext = frame_img_name_no_ext[6:]   # frame_
        img_name = img_name_no_ext + '.png'

        transforms_train = transforms_train_ed if is_end else transforms_train_st
        transforms_train['frames'].append(
            {
                'file_path': './train/' + img_name_no_ext,
                'rotation': 0.0,
                'transform_matrix': [transform_info['cam_in_ob'][i:i+4] for i in range(0, 16, 4)]
            }
        )

        source_img = os.path.join(path, 'color_rgba', img_name)
        target_img = os.path.join(path, 'end' if is_end else 'start', 'train', img_name)
        Image.open(source_img).save(target_img)
        source_img = os.path.join(path, 'depth_filtered', img_name)
        target_img = os.path.join(path, 'end' if is_end else 'start', 'train_d', img_name)
        Image.open(source_img).save(target_img)
        print('done with:', img_name)

    save_dict_to_json(transforms_train_st, os.path.join(start_path, 'transforms_train.json'))
    save_dict_to_json(transforms_train_ed, os.path.join(end_path, 'transforms_train.json'))
    save_dict_to_json(transforms_test, os.path.join(start_path, 'transforms_test.json'))
    save_dict_to_json(transforms_test, os.path.join(end_path, 'transforms_test.json'))


if __name__ == '__main__':
    # data_path = 'data/dta/storage_45135'  # False
    # data_path = 'data/dta/USB_100109'  # True
    # data_path = 'data/dta/blade_103706'  # True
    # data_path = 'data/dta_multi/fridge_10489' # False
    data_path = 'data/dta_multi/storage_47254'  # False

    gen_rgba_images(data_path)
    convert_to_nerf(data_path, res_x=800, is_start_zero=False)
