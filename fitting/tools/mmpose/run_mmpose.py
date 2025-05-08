import os
import os.path as osp
from glob import glob
import argparse
import json
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# Function to extract frame index from filename using regex
def extract_frame_idx(filename):
    # Try to match pattern like "results_0001.json"
    match = re.search(r'results_(\d+)\.json', filename)
    if match:
        return int(match.group(1))
    
    # Try to match any number in the filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Default case, just use a sequential index
    return None

args = parse_args()
root_path = args.root_path
print("Root Path: " + root_path)

# Ensure output directory exists and is empty
output_root = os.path.join(args.root_path, 'keypoints_whole_body')
os.system(f'rm -rf {output_root}')
os.makedirs(output_root, exist_ok=True)

# Run mmpose
cmd = 'python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py dw-ll_ucoco_384.pth  --input ' + osp.join(root_path, 'frames') + ' --output-root ' + output_root + ' --save-predictions'
print(cmd)
os.system(cmd)

# Process the prediction files
temp_output_dir = osp.join(root_path, 'temp_keypoints')
os.makedirs(temp_output_dir, exist_ok=True)
output_path_list = glob(osp.join(output_root, '*.json'))

print(f"Found {len(output_path_list)} JSON files")
if len(output_path_list) == 0:
    print("No JSON files found. Check if mmpose command ran successfully.")
    exit(1)

# First check the structure of the filenames
sample_filenames = [osp.basename(path) for path in output_path_list[:5]]

# Track which frame indices we've seen
processed_frames = {}

for i, output_path in enumerate(output_path_list):
    # Extract frame index from filename
    filename = osp.basename(output_path)
    frame_idx = extract_frame_idx(filename)
    
    if frame_idx is None:
        # If we can't extract a frame index, use a sequential number
        frame_idx = i
        print(f"Warning: Could not extract frame index from {filename}, using {frame_idx} instead")
    
    # Check for duplicate frame indices
    if frame_idx in processed_frames:
        print(f"Warning: Duplicate frame index {frame_idx} found in {filename} and {processed_frames[frame_idx]}")
    
    processed_frames[frame_idx] = filename
    
    try:
        with open(output_path) as f:
            out = json.load(f)
        
        kpt_save = None
        for i in range(len(out['instance_info'])):
            xy = np.array(out['instance_info'][i]['keypoints'], dtype=np.float32).reshape(-1,2)
            score = np.array(out['instance_info'][i]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
            kpt = np.concatenate((xy, score),1) # x, y, score
            if (kpt_save is None) or (kpt_save[:,2].mean() < kpt[:,2].mean()):
                kpt_save = kpt
        
        # Save with zero-padded filename (0000.json, 0001.json, etc.)
        output_filename = f'{frame_idx:04d}.json'
        with open(osp.join(temp_output_dir, output_filename), 'w') as f:
            json.dump(kpt_save.tolist(), f)
            
    except Exception as e:
        print(f"Error processing {output_path}: {str(e)}")

# Clean up and rename the output directory
os.system(f'rm -rf {output_root}')
os.system(f'mv {temp_output_dir} {output_root}')

print(f"Done! {len(processed_frames)} keypoint files saved to {output_root} with zero-padded filenames.")