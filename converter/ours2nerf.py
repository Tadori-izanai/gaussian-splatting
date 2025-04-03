import os
import shutil
from pathlib import Path

from converter.artgs2nerf import move_dir

def find_files_with_suffix(directory, suffix):
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            matching_files.append(filename)
    return matching_files

def convert_to_nerf(path: str) -> None:
    gt = os.path.join(path, 'gt')
    os.makedirs(gt, exist_ok=True)

    move_dir(os.path.join(path, 'images'), os.path.join(gt, 'images'), rm=False)
    for state in ['start', 'end']:
        gt_state = os.path.join(gt, state)
        curr = os.path.join(path, state)

        for dirname in [f'{state}_objs', f'{state}_objs-u']:
            move_dir(os.path.join(curr, dirname), os.path.join(gt_state, dirname), rm=True)
        for filename in [
            f'{state}.obj', f'{state}.obj.mtl',
            f'{state}_rotate.ply', f'{state}_static_rotate.ply', f'{state}_dynamic_rotate.ply'
        ]:
            shutil.move(os.path.join(curr, filename), os.path.join(gt_state, filename))

        dynamic_dir = os.path.join(curr, f'{state}_dynamic_rotates')
        for filename in find_files_with_suffix(dynamic_dir, '.ply'):
            shutil.move(
                os.path.join(dynamic_dir, filename),
                os.path.join(gt_state, f'{state}_dynamic_{Path(filename).stem}_rotate.ply')
            )
        shutil.rmtree(dynamic_dir)

if __name__ == '__main__':
    # data = '../data/teeburu34178'
    data = '../data/teeburu34610'
    convert_to_nerf(data)

    pass
