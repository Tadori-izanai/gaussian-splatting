import os
import sys
import numpy as np
import argparse

# Assuming the script is run from the project root, we can import from the project
from scene.dataset_readers import readCamerasFromTransforms
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

def check_visibility_for_camera(points, cam_info, epsilon):
    """
    Checks visibility of points for a single camera and returns a boolean mask.
    This logic is based on the functions in your project.
    """
    if cam_info.image_d is None:
        return np.zeros(len(points), dtype=bool)

    # Replicate the coordinate system transformations from your project's functions
    # to get the world-to-camera matrix in the Blender convention.
    w2c_colmap = getWorld2View2(cam_info.R, cam_info.T)
    c2w_colmap = np.linalg.inv(w2c_colmap)
    c2w_blender = c2w_colmap.copy()
    c2w_blender[:3, 1:3] *= -1  # Convert C2W from COLMAP to Blender convention
    w2c_blender = np.linalg.inv(c2w_blender) # Blender convention world-to-camera

    # Transform points to Blender camera space (homogeneous coordinates)
    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam_homo = (w2c_blender @ points_homo.T).T

    # In Blender camera space, points in front of the camera have a negative Z value.
    in_front_mask = points_cam_homo[:, 2] < 0

    # Get the projection matrix (Blender convention)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=cam_info.FovX, fovY=cam_info.FovY).cpu().numpy()

    # Project points to clip space
    points_clip_homo = (proj_matrix @ points_cam_homo.T).T
    
    # Perform perspective division, avoiding division by zero
    w = points_clip_homo[:, 3]
    w[np.abs(w) < 1e-7] = 1e-7
    points_ndc = points_clip_homo[:, :3] / w[:, np.newaxis]

    # Convert from NDC to pixel coordinates (Y-down)
    x_px = ((points_ndc[:, 0] + 1) * cam_info.width / 2)
    y_px = ((1 - points_ndc[:, 1]) * cam_info.height / 2)

    # Create a mask for points that project inside the image boundaries
    in_bounds_mask = (x_px >= 0) & (x_px < cam_info.width) & (y_px >= 0) & (y_px < cam_info.height)
    
    # The final mask for points that can be checked against the depth map
    valid_proj_mask = in_front_mask & in_bounds_mask
    
    if not np.any(valid_proj_mask):
        return np.zeros(len(points), dtype=bool)

    # Get integer pixel coordinates for valid points to sample the depth map
    x_px_valid = x_px[valid_proj_mask].astype(int)
    y_px_valid = y_px[valid_proj_mask].astype(int)
    
    # Calculated depth is the positive distance along the camera's local Z axis
    calculated_depths = -points_cam_homo[valid_proj_mask, 2]
    
    # Sample the depth from the depth map at the projected locations
    sampled_depths = cam_info.image_d[y_px_valid, x_px_valid]

    # A point is visible if its depth is less than or equal to the sampled depth (with tolerance)
    visible_in_cam = calculated_depths <= (sampled_depths + epsilon)
    
    # Map the visibility results from the subset of valid points back to the original N-point array
    camera_visibility_mask = np.zeros(len(points), dtype=bool)
    camera_visibility_mask[valid_proj_mask] = visible_in_cam
    
    return camera_visibility_mask

def get_visibility_mask(points_to_check: np.ndarray, data_dir: str, epsilon: float = 0.005) -> np.ndarray:
    """
    Calculates the visibility of a point cloud against a set of depth images.

    Args:
        points_to_check (np.ndarray): The (N, 3) point cloud to check.
        data_dir (str): Path to the object's data directory (e.g., 'data/teeburu23372').
        epsilon (float): Tolerance for depth comparison.

    Returns:
        np.ndarray: A boolean mask of shape (N,) where True means a point is visible.
    """
    num_points = len(points_to_check)
    print(f"Checking visibility for {num_points} points...")

    # 1. Load camera and depth data
    camera_data_path = os.path.join(os.path.realpath(data_dir), 'end')
    transforms_file = os.path.join(camera_data_path, 'transforms_train.json')
    
    if not os.path.exists(transforms_file):
        print(f"Error: Transforms file not found at '{transforms_file}'")
        return np.zeros(num_points, dtype=bool)
        
    try:
        print(f"Loading cameras from {transforms_file}...")
        cam_infos = readCamerasFromTransforms(camera_data_path, "transforms_train.json", white_background=False, extension=".png")
        print(f"Loaded {len(cam_infos)} cameras.")
    except Exception as e:
        print(f"Error loading camera info: {e}")
        return np.zeros(num_points, dtype=bool)

    # 2. Perform visibility check across all cameras
    final_visibility_mask = np.zeros(num_points, dtype=bool)

    for i, cam in enumerate(cam_infos):
        print(f"\r- Checking visibility in camera {i+1}/{len(cam_infos)}...", end="")
        camera_mask = check_visibility_for_camera(points_to_check, cam, epsilon)
        final_visibility_mask |= camera_mask

    visible_count = np.sum(final_visibility_mask)
    print(f"\nProcessing complete. {visible_count} out of {num_points} points are visible.")
    
    return final_visibility_mask

if __name__ == '__main__':
    # This is an example of how to use the get_visibility_mask function.
    # It requires a sample data directory and a point cloud file.
    parser = argparse.ArgumentParser(description="Example usage for get_visibility_mask function.")
    parser.add_argument("--data_dir", type=str, default="data/teeburu23372", help="Path to the object's data directory.")
    parser.add_argument("--pcd_file", type=str, default="example_pcd.npy", help="Path to the .npy file containing the point cloud (Nx3).")
    parser.add_argument("--output_file", type=str, default="visibility_mask_example.npy", help="Path to save the output visibility mask .npy file.")
    
    args = parser.parse_args()

    print("Running example usage...")

    # Check if the example data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Example data directory '{args.data_dir}' not found. Skipping example.")
        sys.exit(0)
        
    # Create a dummy point cloud file for the example if it doesn't exist
    if not os.path.exists(args.pcd_file):
        print(f"Creating a dummy point cloud file for demonstration: '{args.pcd_file}'")
        # Create 100 random points within a plausible bounding box for the scene
        dummy_points = np.random.rand(100, 3) * 2 - 1 
        np.save(args.pcd_file, dummy_points)

    print(f"Using data directory: '{args.data_dir}'")
    print(f"Using point cloud: '{args.pcd_file}'")

    # Load the points and run the main function
    points = np.load(args.pcd_file)
    visibility_mask = get_visibility_mask(points, args.data_dir)
    
    # Save the output
    np.save(args.output_file, visibility_mask)
    print(f"Example visibility mask saved to '{args.output_file}'")
