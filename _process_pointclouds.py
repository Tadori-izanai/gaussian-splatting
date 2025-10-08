import os
import shutil
import glob

def process_pointclouds(data_dir='data/artgs', dest_dir='pointclouds'):
    """
    Copies and renames point cloud files from a nested directory structure
    to a flattened destination directory.
    """
    # Ensure the destination directory exists
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Destination directory '{dest_dir}' is ready.")
    except OSError as e:
        print(f"Error: Could not create destination directory {dest_dir}: {e}")
        return

    # Check if the data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Source directory '{data_dir}' not found.")
        return

    # Get all subdirectories in the data directory
    subdirectories = [f for f in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(f)]

    if not subdirectories:
        print(f"No subdirectories found in '{data_dir}'.")
        return

    print(f"Found {len(subdirectories)} potential data directories to process...")

    processed_count = 0
    for subdir in subdirectories:
        dir_name = os.path.basename(subdir)
        print(f"\nProcessing '{dir_name}'...")

        # Define the expected source files and their corresponding destination names
        file_map = {
            os.path.join(subdir, 'start', 'points3d.ply'): os.path.join(dest_dir, f'{dir_name}_start.ply'),
            os.path.join(subdir, 'start', 'points3d-100k.ply'): os.path.join(dest_dir, f'{dir_name}_start-100k.ply'),
            os.path.join(subdir, 'end', 'points3d.ply'): os.path.join(dest_dir, f'{dir_name}_end.ply'),
            os.path.join(subdir, 'end', 'points3d-100k.ply'): os.path.join(dest_dir, f'{dir_name}_end-100k.ply')
        }

        # Check if all source files exist before attempting to copy
        all_files_present = True
        for src_path in file_map.keys():
            if not os.path.exists(src_path):
                print(f"  - Skipping: Source file not found -> {src_path}")
                all_files_present = False
                break
        
        if not all_files_present:
            continue

        # If all files are present, copy and rename them
        try:
            for src_path, dest_path in file_map.items():
                shutil.copy2(src_path, dest_path)
                print(f"  - Copied '{src_path}' to '{dest_path}'")
            processed_count += 1
        except Exception as e:
            print(f"  - An error occurred while processing '{dir_name}': {e}")

    print(f"\nDone. Successfully processed {processed_count} out of {len(subdirectories)} directories.")

if __name__ == "__main__":
    process_pointclouds()
