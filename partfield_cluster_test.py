from os.path import join as pjoin
import shutil

import numpy as np

from main import *
from _get_visibility_mask import get_visibility_mask

def find_files_with_suffix(directory, suffix):
    if not os.path.exists(directory):
        return []
    matching_files = []
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            matching_files.append(filename)
    return matching_files

def dbscan_filter(x, eps=dbs_eps):
    clustering = DBSCAN(eps=eps, min_samples=1).fit(x)

    labels = clustering.labels_
    mask_with_pts = sorted(
        [(labels == k, x[labels == k]) for k in np.unique_values(labels)],
        key=lambda item: len(item[1]), reverse=True
    )
    mask, _ = mask_with_pts[0]
    return mask

def split_point_cloud_by_color(ply_path):
    """
    Reads a PLY file, finds unique colors, and saves a separate PLY file
    for each color group into a new directory.
    """
    # 1. Validate input path
    if not os.path.exists(ply_path):
        print(f"Error: Input file not found at '{ply_path}'")
        return

    # 2. Create an output directory named after the ply file
    output_dir = os.path.splitext(os.path.basename(ply_path))[0]
    output_dir = pjoin(os.path.dirname(ply_path), output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output will be saved in directory: '{output_dir}'")
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return

    # 3. Load the point cloud using the now-robust fetchPly
    print(f"Loading point cloud from '{ply_path}'...")
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"An error occurred while loading the PLY file with fetchPly: {e}")
        return

    # fetchPly returns colors as floats (0-1). Convert to uint8 (0-255) for robust grouping.
    colors_uint8 = (pcd.colors * 255).astype(np.uint8)

    # 4. Find unique colors
    unique_colors = np.unique(colors_uint8, axis=0)
    print(f"Found {len(unique_colors)} unique colors.")

    # 5. Iterate, filter, and save for each unique color
    for i, color in enumerate(unique_colors):
        # Create a boolean mask for the current color
        mask = np.all(colors_uint8 == color, axis=1)

        # Apply the mask to get the subset of data
        subset_points = pcd.points[mask]
        subset_colors = colors_uint8[mask]  # Already in uint8 format
        subset_normals = pcd.normals[mask]

        db_mask = dbscan_filter(subset_points)
        subset_points = subset_points[db_mask]
        subset_colors = subset_colors[db_mask]
        subset_normals = subset_normals[db_mask]

        # Define the output path for the new PLY file
        output_path = os.path.join(output_dir, f"{i}.ply")

        print(f"  - Saving {len(subset_points)} points with color {color} to '{output_path}'")

        # Save the new point cloud using the storePly function
        storePly(output_path, subset_points, subset_colors, subset_normals)
    print("\nProcessing complete.")

def get_mask(x: torch.tensor, y: torch.tensor, thr: int, ply_path=None) -> torch.tensor:
    cd = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, single_directional=True)[0][0]
    cd /= torch.max(cd)
    cd_is = inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6))
    if ply_path is not None:
        plot_hist(cd_is, os.path.join(ply_path, 'cd_is-1m.png'))

    mask = (cd_is > thr)
    return mask

def filter_clusters(out_path: str, factor=5) -> int:
    source_dir = pjoin(out_path, 'points3d')
    target_dir = pjoin(out_path, 'clusters')
    os.makedirs(target_dir, exist_ok=True)

    pcd_sizes = {}
    max_size = 0
    for filename in find_files_with_suffix(source_dir, '.ply'):
        pcd = fetchPly(pjoin(source_dir, filename))
        sz = len(pcd.points)
        max_size = max(max_size, sz)
        pcd_sizes[filename] = sz

    cnt = 0
    for filename, sz in pcd_sizes.items():
        if sz >= max_size // factor:
            shutil.copy(pjoin(source_dir, filename), pjoin(target_dir, 'points3d_' + filename))
            cnt += 1

    print(cnt)
    return cnt

def est_normals(ply_path: str):
    for filename in find_files_with_suffix(ply_path, '.ply'):
        pcd = fetchPly(pjoin(ply_path, filename))
        if np.linalg.norm(pcd.normals[0]) > .1:
            continue
        print('estimating', filename)
        normals = estimate_normals_o3d(pcd.points)
        storePly(os.path.join(ply_path, filename),
                 pcd.points, (pcd.colors * 255).astype(np.uint8), normals)
    #return

def partfield_seg(out_path: str, data_path: str, num_movable: int, thr: int=-5):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    ply_path = os.path.join(out_path, 'clustering')
    os.makedirs(ply_path, exist_ok=True)
    os.makedirs(os.path.join(ply_path, 'clusters'), exist_ok=True)

    st_data = pjoin(os.path.realpath(data_path), 'start_0_19.ply')
    # st_data = pjoin(os.path.realpath(data_path), 'start_0_12.ply')
    ed_data = pjoin(os.path.realpath(data_path), 'end_0_19.ply')
    st_pcd, ed_pcd = fetchPly(st_data), fetchPly(ed_data)
    xyz_st, xyz_ed = np.asarray(st_pcd.points), np.asarray(ed_pcd.points)

    x = torch.tensor(xyz_st, device='cuda').unsqueeze(0)
    y = torch.tensor(xyz_ed, device='cuda').unsqueeze(0)

    mask = get_mask(x, y, thr, ply_path).detach().cpu().numpy()

    print(mask.shape)
    print(st_pcd.colors.shape)

    x = x[0].detach().cpu().numpy()[mask]
    storePly(pjoin(ply_path, f'points3d.ply'), x,
             (st_pcd.colors * 255).astype(np.uint8)[mask], st_pcd.normals[mask])

    # vis = get_visibility_mask(x, data_path)
    # storePly(pjoin(ply_path, f'points3d-vis.ply'), x[vis],
    #          (st_pcd.colors * 255).astype(np.uint8)[mask][vis], st_pcd.normals[mask][vis])

    # seg
    split_point_cloud_by_color(pjoin(ply_path, 'points3d.ply'))
    _ = filter_clusters(ply_path, factor=5)
    est_normals(pjoin(ply_path, 'clusters'))

if __name__ == '__main__':
    # K = 4
    # st = 'output/single/tbr4_st'
    # ed = 'output/single/tbr4_ed'
    # data = 'data/teeburu34178'
    # out = 'output/tbr4'
    # rev = False

    K = 10
    st = 'output/single/str_st'
    ed = 'output/single/str_ed'
    data = 'data/sutoreeji47585'
    out = 'output/str'
    rev = False
    #
    # cd_thr = -10
    # dbs_eps = 0.004

    # K = 5
    # st = 'output/single/tbr5_st'
    # ed = 'output/single/tbr5_ed'
    # data = 'data/teeburu34610'
    # out = 'output/tbr5'
    # rev = False

    partfield_seg(out, data, K, thr=cd_thr)

    pass
