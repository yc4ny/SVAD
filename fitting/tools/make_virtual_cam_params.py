import os
import os.path as osp
import json
import cv2
import shutil
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--output_dir', type=str, dest='output_dir', help="Output directory for camera parameters")
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

args = parse_args()
root_path = args.root_path

# Get image dimensions
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
if not img_path_list:
    print(f"ERROR: No PNG files found in frames directory: {osp.join(root_path, 'frames')}")
    exit(1)

first_img = cv2.imread(img_path_list[0])
if first_img is None:
    print(f"ERROR: Could not read image: {img_path_list[0]}")
    exit(1)

img_height, img_width = first_img.shape[:2]

secondary_save_path = osp.join(root_path, 'cam_params')
os.makedirs(secondary_save_path, exist_ok=True)
print(f"Also saving camera parameters to secondary location: {secondary_save_path}")

# Extract frame indices from filenames
frame_idx_list = []
for img_path in img_path_list:
    try:
        frame_idx = int(osp.basename(img_path).split('.')[0])
        frame_idx_list.append(frame_idx)
    except ValueError:
        print(f"Warning: Could not parse frame index from {img_path}")

frame_idx_list.sort()
print(f"Processing {len(frame_idx_list)} frames")

# Generate camera parameters for each frame and save to both locations
for frame_idx in frame_idx_list:
    cam_params = {
        'R': np.eye(3).astype(np.float32).tolist(),
        't': np.zeros((3), dtype=np.float32).tolist(),
        'focal': (2000, 2000),
        'princpt': (img_width/2, img_height/2)
    }
    
    secondary_output_file = osp.join(secondary_save_path, f'{frame_idx:04d}.json')
    with open(secondary_output_file, 'w') as f:
        json.dump(cam_params, f)

print(f"Created camera parameters for {len(frame_idx_list)} frames in both locations")
print("Camera parameter generation complete")