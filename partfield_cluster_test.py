from os.path import join as pjoin
import shutil

import numpy as np

from main import *
from _get_visibility_mask import get_visibility_mask

def obbs_to_ply(axes: np.ndarray, centers: np.ndarray, extents: np.ndarray, directory: str):
    """
    将 K 个 OBB (Oriented Bounding Boxes) 转换为 K 个 PLY 格式的 mesh 文件并保存。

    参数:
        axes (np.ndarray): Shape (K, 3, 3). K 个 OBB 的旋转矩阵（方向轴）。
        centers (np.ndarray): Shape (K, 3). K 个 OBB 的中心点坐标。
        extents (np.ndarray): Shape (K, 3). K 个 OBB 在其局部坐标系下的半长（x, y, z方向）。
        dir (str): 用于保存生成的 .ply 文件的目录路径。
    """
    # 确保输出目录存在
    os.makedirs(directory, exist_ok=True)

    # 1. 定义一个单位立方体的顶点和面
    # 单位立方体的8个顶点，范围从-1到1
    unit_vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float64)

    # 单位立方体的12个三角面片 (6个四边形面 -> 12个三角形)
    # 顶点的索引顺序确保了法线朝外
    faces = np.array([
        [0, 3, 2], [0, 2, 1],  # bottom (-z)
        [4, 5, 6], [4, 6, 7],  # top (+z)
        [0, 1, 5], [0, 5, 4],  # front (-y)
        [2, 3, 7], [2, 7, 6],  # back (+y)
        [0, 4, 7], [0, 7, 3],  # left (-x)
        [1, 2, 6], [1, 6, 5]  # right (+x)
    ])

    # 获取 OBB 的数量 K
    k = centers.shape[0]

    # 2. 遍历每个 OBB 并生成对应的 PLY 文件
    for i in range(k):
        # 获取当前 OBB 的参数
        R = axes[i]
        c = centers[i]
        e = extents[i]

        # 3. 对单位立方体的顶点进行变换
        # a. 缩放: 根据 extents 拉伸顶点
        scaled_vertices = unit_vertices * e

        # b. 旋转: 应用旋转矩阵
        #    注意：我们处理的是行向量顶点，所以顶点要右乘旋转矩阵的转置 (v' = v @ R.T)
        rotated_vertices = scaled_vertices @ R.T

        # c. 平移: 移动到指定的中心点
        transformed_vertices = rotated_vertices + c

        # 4. 准备写入 PLY 文件
        ply_path = os.path.join(directory, f"obb_{i}.ply")

        with open(ply_path, 'w') as f:
            # 写入 PLY 文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(transformed_vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # 写入顶点数据
            for vertex in transformed_vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

            # 写入面数据
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"成功在目录 '{directory}' 中生成了 {k} 个 PLY 文件。")

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

    masks = []
    for i in range(3):
        if i < len(mask_with_pts):
            mask, _ = mask_with_pts[i]
            masks.append(mask)
    return masks

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
        pcd = fetchPly(ply_path, uint_colors=True)
    except Exception as e:
        print(f"An error occurred while loading the PLY file with fetchPly: {e}")
        return

    # fetchPly returns colors as floats (0-1). Convert to uint8 (0-255) for robust grouping.
    colors_uint8 = pcd.colors

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

        db_masks = dbscan_filter(subset_points)
        for j, db_mask in enumerate(db_masks):
            sub_subset_points = subset_points[db_mask]
            sub_subset_colors = subset_colors[db_mask]
            sub_subset_normals = subset_normals[db_mask]

            # Define the output path for the new PLY file
            output_path = os.path.join(output_dir, f"{i}_{j}.ply")

            print(f"  - Saving {len(sub_subset_points)} points with color {color} to '{output_path}'")
            storePly(output_path, sub_subset_points, sub_subset_colors, sub_subset_normals)

    print("\nProcessing complete.")

def eval_mask_by_cd(x: torch.tensor, y: torch.tensor, thr: int, ply_path=None) -> torch.tensor:
    cd = chamfer_distance(x, y, batch_reduction=None, point_reduction=None, single_directional=True)[0][0]
    cd /= torch.max(cd)
    cd_is = inverse_sigmoid(torch.clamp(cd, 1e-6, 1 - 1e-6))
    if ply_path is not None:
        plot_hist(cd_is, os.path.join(ply_path, 'cd_is-1m.png'))

    mask = (cd_is > thr)
    return mask

def filter_clusters(out_path: str, factor=5) -> int:
    source_dir = pjoin(out_path, 'points3d')
    target_dir = pjoin(out_path, 'clusters-unmerged')
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
            print('saved:', filename)
            shutil.copy(pjoin(source_dir, filename), pjoin(target_dir, f'points3d_{cnt}.ply'))
            cnt += 1

    print('# total clusters:', cnt)
    return cnt

def partfield_seg(out_path: str, data_path: str, thr: int=-5):
    mk_output_dir(out_path, os.path.join(data_path, 'start'))
    ply_path = os.path.join(out_path, 'clustering')
    os.makedirs(ply_path, exist_ok=True)
    os.makedirs(os.path.join(ply_path, 'clusters-unmerged'), exist_ok=True)

    st_data = pjoin(os.path.realpath(data_path), 'start_0_19.ply')
    ed_data = pjoin(os.path.realpath(data_path), 'end/points3d.ply')
    st_pcd, ed_pcd = fetchPly(st_data, uint_colors=True), fetchPly(ed_data, uint_colors=True)

    print('estimating normals ...')
    normals_st = estimate_normals_o3d(st_pcd.points)

    xyz_st, xyz_ed = np.asarray(st_pcd.points), np.asarray(ed_pcd.points)
    x = torch.tensor(xyz_st, device='cuda', dtype=torch.float).unsqueeze(0)
    y = torch.tensor(xyz_ed, device='cuda', dtype=torch.float).unsqueeze(0)

    mask = eval_mask_by_cd(x, y, thr, ply_path).detach().cpu().numpy()

    x = x[0].detach().cpu().numpy()
    storePly(pjoin(ply_path, f'points3d.ply'), x[mask], st_pcd.colors[mask], normals_st[mask])

    # seg
    split_point_cloud_by_color(pjoin(ply_path, 'points3d.ply'))
    _ = filter_clusters(ply_path, factor=10)

def pre_merge(out_path: str):
    target_dir = pjoin(out_path, 'clustering/clusters')
    os.makedirs(target_dir, exist_ok=True)
    nrm_dir = pjoin(out_path, 'clustering/nrm_axes')
    os.makedirs(nrm_dir, exist_ok=True)

    source_dir = pjoin(out_path, 'clustering/clusters-unmerged')
    ply_files = find_files_with_suffix(source_dir, '.ply')
    num_files = len(ply_files)
    print(ply_files)

    axes = np.zeros((num_files, 3, 3))
    parts = []
    pcds = []
    for k, filename in enumerate(ply_files):
        pcd = fetchPly(pjoin(source_dir, filename), uint_colors=True)
        axes[k] = estimate_principal_directions(np.asarray(pcd.normals), ort='gs')
        pcds.append(pcd)
        parts.append(np.asarray(pcd.points))

    _ = find_and_unify_orthogonal(axes)
    # print(axes)

    bb_centers, bb_extents = np.zeros((num_files, 3)), np.zeros((num_files, 3))
    for k, pts in enumerate(parts):
        center, extent = get_bounding_box(pts, axes[k])
        bb_centers[k] = center
        bb_extents[k] = extent

        # o = np.zeros(3)
        # dirs = axes[k]
        # for i in np.arange(3):
        #     d = -dirs[i] if (dirs[i] @ center < 0) else dirs[i]
        #     save_axis_mesh(d, o, pjoin(nrm_dir, f'axis{k}_{i}.ply'), center)

    obbs_to_ply(axes, bb_centers, bb_extents, nrm_dir)

    proximity_matrix, merge_likelihood = get_obb_proximity_matrix(axes, bb_centers, bb_extents * 1.05)
    print(proximity_matrix)
    print(merge_likelihood)

    ds = DisjointSet(num_files)
    for k in range(num_files):
        if merge_likelihood[k] < 0.2:
            continue
        j = np.argmax(proximity_matrix[k])
        if not ds.is_connected(k, j):
            ds.connect(k, j)
            print(f'pre-merged: {ply_files[k]} to {ply_files[j]}')

    uniques = np.unique(ds.parent)
    for idx in uniques:
        indices = [k for k in range(num_files) if ds.parent[k] == idx]
        points = np.concatenate([np.asarray(pcds[k].points) for k in indices], axis=0)
        normals = np.concatenate([np.asarray(pcds[k].normals) for k in indices], axis=0)
        colors = np.tile(pcds[idx].colors[0], (len(points), 1))
        output_path = pjoin(target_dir, ply_files[idx])
        storePly(output_path, points, colors, normals)

if __name__ == '__main__':
    if True:
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
        pass
    # fi

    # ArtGS (5)
    if True:
        # K = 3
        # st = 'output/artgs/oven_st'
        # ed = 'output/artgs/oven_ed'
        # data = 'data/artgs/oven_101908'
        # out = 'output/artgs/oven'
        # rev = False

        # K = 3
        # st = 'output/artgs/tbl3_st'
        # ed = 'output/artgs/tbl3_ed'
        # data = 'data/artgs/table_25493'
        # out = 'output/artgs/tbl3'
        # rev = True

        # K = 3
        # st = 'output/artgs/sto3_st'
        # ed = 'output/artgs/sto3_ed'
        # data = 'data/artgs/storage_45503'
        # out = 'output/artgs/sto3'
        # rev = True

        # K = 4
        # st = 'output/artgs/tbl4_st'
        # ed = 'output/artgs/tbl4_ed'
        # data = 'data/artgs/table_31249'
        # out = 'output/artgs/tbl4'
        # rev = False

        # K = 6
        # st = 'output/artgs/sto6_st'
        # ed = 'output/artgs/sto6_ed'
        # data = 'data/artgs/storage_47648'
        # out = 'output/artgs/sto6'
        # rev = True

        pass
    # fi

    # partfield_seg(out, data, thr=cd_thr)
    pre_merge(out)

    pass
