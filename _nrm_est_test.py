from os.path import join as pjoin

from main import *

def load_json_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as json_file:
        info = json.load(json_file)
    return info

def jitter(pts: np.ndarray, factor: float) -> np.ndarray:
    np.random.seed(42)

    # 1. Calculate the overall scale of the point cloud
    # We use the peak-to-peak (max-min) range along each axis
    extent = np.ptp(pts, axis=0)

    # The characteristic scale is the average of the extents
    scale = np.mean(extent)

    # 2. Generate random Gaussian noise
    # The noise is scaled by our characteristic scale and the user-provided factor
    noise = np.random.randn(*pts.shape) * (factor * scale)

    # 3. Add the noise to the original points
    jittered_points = pts + noise
    return jittered_points

def eval_normals(num_movable: int, path: str, method: str):
    fstr = '' if f == 0 else f'_{f}'
    out_path = pjoin(path, f'clustering/test/{method}{fstr}')
    os.makedirs(out_path, exist_ok=True)

    pts = list_of_clusters(pjoin(path, 'clustering/clusters'), num_movable)
    mu = np.load(pjoin(path, 'mu_init.npy'))

    for i in np.arange(len(pts)):
        pts[i] = jitter(pts[i], f)

    angles = np.zeros((num_movable, 3))

    def process_pcd(k, pcd):
        print('processing cluster', k)
        ply_path = pjoin(out_path, f'pts_{k}.ply')
        if os.path.exists(ply_path):
            normals = fetchPly(ply_path).normals
        else:
            normals = methods[method](pcd)
            storePly(ply_path, pcd, np.zeros_like(pcd), normals)

        axes = estimate_principal_directions(normals, ort='gs')  # 3,3
        o = np.zeros(3)
        for i in np.arange(3):
            d = -axes[i] if (axes[i] @ mu[k] < 0) else axes[i]
            save_axis_mesh(d, o, os.path.join(out_path, f'axis{k}_{i}.ply'), mu[k])

        if f == 0:
            np.save(pjoin(out_path, f'angle_{k}.npy'), axes)
        else:
            gt_axes = np.load(pjoin(path, f'clustering/test/{method}/angle_{k}.npy'))
            dots = np.clip(np.sum(axes * gt_axes, axis=1), -1.0, 1.0)
            angle = np.acos(np.abs(dots)).min()
            angle = np.rad2deg(angle)
            angles[k] = angle
        #fi

    for kk, pp in enumerate(pts):
        process_pcd(kk, pp)

    if f != 0:
        print('angles:')
        print(angles)
        print('mean:')
        print(angles.mean(axis=1))
    #fi

methods = {
    'pca': estimate_normals_o3d,
    'ransac': estimate_normals_ransac_o3d,
}
f = 0.005

if __name__ == '__main__':
    K, out = 4, 'output/tbr4'

    # eval_normals(K, out, 'pca')
    eval_normals(K, out, 'ransac')

    pass
