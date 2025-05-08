import os
import os.path as osp
import json
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import torch
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--output_dir', type=str, dest='output_dir', help="Custom output directory")
    parser.add_argument('--gpu', type=str, dest='gpu_ids', help="GPU ID to use")
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# Helper function to safely extract frame index from filename
def extract_frame_idx(filename):
    # Remove path and get just the filename
    base_filename = os.path.basename(filename)
    
    # For JSON files like "0000.json"
    if base_filename.endswith('.json'):
        try:
            # Extract everything before .json and convert to int
            return int(base_filename[:-5])
        except ValueError:
            # If it's not a simple number, try to find the numeric part
            match = re.search(r'(\d+)', base_filename)
            if match:
                return int(match.group(1))
    
    # For other file types, try to extract any numeric part
    match = re.search(r'(\d+)', base_filename)
    if match:
        return int(match.group(1))
    
    # If all else fails
    raise ValueError(f"Could not extract frame index from filename: {filename}")

# Validate paths and directory structure
def validate_paths(root_path):
    frames_path = osp.join(root_path, 'frames')
    
    # Check if the root path exists
    if not osp.exists(root_path):
        print(f"ERROR: Root path does not exist: {root_path}")
        return False
        
    # Check if the frames directory exists
    if not osp.exists(frames_path):
        print(f"ERROR: Frames directory does not exist: {frames_path}")
        print(f"Attempting to create frames directory...")
        try:
            os.makedirs(frames_path, exist_ok=True)
            print(f"Created frames directory: {frames_path}")
        except Exception as e:
            print(f"Failed to create frames directory: {e}")
            return False
    
    # Check if frames directory has image files
    image_files = glob(osp.join(frames_path, '*.png')) + glob(osp.join(frames_path, '*.jpg'))
    if not image_files:
        print(f"WARNING: No image files found in frames directory: {frames_path}")
        print(f"Make sure there are .png or .jpg files in this directory.")
        return False

    return True

def main():
    args = parse_args()
    root_path = args.root_path
    
    # Set CUDA_VISIBLE_DEVICES if GPU ID is provided
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using GPU: {args.gpu_ids}")
    
    print(f"Validating paths for root: {root_path}")
    if not validate_paths(root_path):
        print("Path validation failed. Exiting.")
        sys.exit(1)
    
    # Create necessary directories
    if args.output_dir:
        output_save_path = args.output_dir
    else:
        output_save_path = './flame_parmas_out'
    
    print(f"Removing and recreating output directory: {output_save_path}")
    os.makedirs(output_save_path, exist_ok=True)
    
    # run DECA with full path information
    frames_path = osp.abspath(osp.join(root_path, 'frames'))
    output_save_path_abs = osp.abspath(output_save_path)
    print(f"Running DECA on frames in: {frames_path}")
    print(f"Saving output to: {output_save_path_abs}")
    
    # Modified command to skip visualization by using --saveVis False
    cmd = f'python demos/demo_reconstruct.py -i "{frames_path}" --saveDepth True --saveObj True --saveVis False --rasterizer_type=pytorch3d --savefolder "{output_save_path_abs}"'
    print(f"Executing command: {cmd}")
    result = os.system(cmd)
    if result != 0:
        print(f"ERROR: DECA command failed with exit code {result}")
        sys.exit(1)
    
    # Create subject-specific flame_init directory
    save_path = osp.join(root_path, 'flame_init', 'flame_params')
    os.makedirs(save_path, exist_ok=True)
    flame_shape_param = []
    
    output_path_list = glob(osp.join(output_save_path, '*.json'))
    if not output_path_list:
        print(f"ERROR: No JSON files found in output directory: {output_save_path}")
        sys.exit(1)
        
    print(f"Processing {len(output_path_list)} JSON files from DECA output")
    for output_path in tqdm(output_path_list):
        try:
            # Use safer frame index extraction
            frame_idx = extract_frame_idx(output_path)
            
            with open(output_path) as f:
                flame_param = json.load(f)
            if flame_param['is_valid']:
                root_pose, jaw_pose = torch.FloatTensor(flame_param['pose'])[:,:3].view(3), torch.FloatTensor(flame_param['pose'])[:,3:].view(3)
                shape = torch.FloatTensor(flame_param['shape']).view(-1)
                expr = torch.FloatTensor(flame_param['exp']).view(-1)
                flame_shape_param.append(shape)
    
                root_pose, jaw_pose, shape, expr = root_pose.tolist(), jaw_pose.tolist(), shape.tolist(), expr.tolist()
                neck_pose, leye_pose, reye_pose = [0,0,0], [0,0,0], [0,0,0]
            else:
                root_pose, jaw_pose, neck_pose, leye_pose, reye_pose, expr, shape = None, None, None, None, None, None, None
            flame_param = {'root_pose': root_pose, 'neck_pose': neck_pose, 'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose, 'expr': expr, 'is_valid': flame_param['is_valid']}
            with open(osp.join(save_path, f'{frame_idx:04d}.json'), 'w') as f:
                json.dump(flame_param, f)
        except Exception as e:
            print(f"Error processing {output_path}: {e}")
            continue
    
    # Calculate mean shape parameter if we have any valid parameters
    if flame_shape_param:
        try:
            print(f"Calculating mean shape parameter from {len(flame_shape_param)} valid frames")
            flame_shape_param = torch.stack(flame_shape_param).mean(0).tolist()
            os.makedirs(osp.join(root_path, 'flame_init'), exist_ok=True)
            with open(osp.join(root_path, 'flame_init', 'shape_param.json'), 'w') as f:
                json.dump(flame_shape_param, f)
            print(f"Saved mean shape parameter to: {osp.join(root_path, 'flame_init', 'shape_param.json')}")
        except Exception as e:
            print(f"Error calculating mean shape parameter: {e}")
    else:
        print("WARNING: No valid flame parameters found, could not calculate mean shape")
    
    # Skip the visualization/renders section entirely
    print("DECA processing completed successfully")

if __name__ == "__main__":
    main()