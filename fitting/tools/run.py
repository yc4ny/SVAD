import os
import os.path as osp
from glob import glob
import sys
import argparse
import shutil

# Main execution
def main():
    # get path
    args = parse_args()
    root_path = osp.abspath(args.root_path)
    print(f"Using root path: {root_path}")
    print(f"Using GPU: {args.gpu_ids}")
    
    # Validate root path
    if not validate_path(root_path):
        print("Root path does not exist. Exiting.")
        sys.exit(1)
    
    # Extract subject_id from path
    if root_path.endswith('/'):
        subject_id = root_path.split('/')[-2]
    else:
        subject_id = root_path.split('/')[-1]
    print(f"Subject ID: {subject_id}")
    
    base_output_dir = args.root_path
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Using output directory: {base_output_dir}")
    
    # Validate frames directory
    frames_path = osp.join(root_path, 'frames')
    if not validate_path(frames_path):
        print("Frames directory does not exist. Exiting.")
        sys.exit(1)
    
    # Check for frame_list_all.txt
    frame_list_path = osp.join(root_path, 'frame_list_all.txt')
    if not osp.exists(frame_list_path):
        print(f"Warning: frame_list_all.txt not found at {frame_list_path}")
        print("Creating a default frame_list_all.txt with all frames")
        # Create a default frame_list_all.txt with all available frames
        img_files = sorted(glob(osp.join(frames_path, '*.png')))
        if not img_files:
            print("No image files found in frames directory. Exiting.")
            sys.exit(1)
            
        with open(frame_list_path, 'w') as f:
            for img_path in img_files:
                frame_name = osp.basename(img_path)
                f.write(f"{frame_name}\n")
        print(f"Created frame_list_all.txt with {len(img_files)} frames")
    
    # Read frame list and handle different formats
    with open(frame_list_path) as f:
        lines = f.readlines()
        frame_idx_list = []
        for line in lines:
            line = line.strip()
            if line.endswith('.png'):
                # Extract the numeric part and convert to int
                try:
                    frame_idx = int(line.split('.')[0])
                    frame_idx_list.append(frame_idx)
                except ValueError:
                    print(f"Warning: Could not parse frame index from {line}")
            else:
                # Try to directly convert to int if no extension
                try:
                    frame_idx = int(line)
                    frame_idx_list.append(frame_idx)
                except ValueError:
                    print(f"Warning: Could not parse frame index from {line}")
    
    if not frame_idx_list:
        print("No valid frame indices found in frame_list_all.txt. Exiting.")
        sys.exit(1)
    
    print(f"Found {len(frame_idx_list)} valid frames")
    
    # remove unnecessary frames
    img_path_list = glob(osp.join(frames_path, '*.png'))
    for img_path in img_path_list:
        try:
            frame_idx = int(osp.basename(img_path).split('.')[0])
            if frame_idx not in frame_idx_list:
                print(f"Removing unnecessary frame: {img_path}")
                os.remove(img_path)
        except (ValueError, OSError) as e:
            print(f"Error processing frame {img_path}: {e}")
    
    # make camera parameters
    if args.use_colmap:
        colmap_dir = './COLMAP'
        if not validate_path(colmap_dir):
            print("COLMAP directory not found. Exiting.")
            sys.exit(1)
            
        os.chdir(colmap_dir)
        if not execute_command(f'python run_colmap.py --root_path "{root_path}" --output_dir "{base_output_dir}/colmap_output" --gpu {args.gpu_ids}', 
                              "COLMAP failed to get camera parameters"):
            sys.exit(1)
        os.chdir('..')
    else:
        if not execute_command(f'python make_virtual_cam_params.py --root_path "{root_path}" --output_dir "{base_output_dir}/cam_params"',
                              "Failed to make virtual camera parameters"):
            sys.exit(1)
    
    # DECA (get initial FLAME parameters)
    deca_dir = './DECA'
    if not validate_path(deca_dir):
        print("DECA directory not found. Exiting.")
        sys.exit(1)
        
    os.chdir(deca_dir)
    if not execute_command(f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python run_deca.py --root_path "{root_path}" --output_dir {os.path.join(base_output_dir, "flame_init")} --gpu {args.gpu_ids}',
                          "DECA failed"):
        sys.exit(1)
        
    os.chdir('..')

    
    # Hand4Whole (get initial SMPLX parameters)
    hand4whole_dir = './Hand4Whole_RELEASE/demo'
    if not validate_path(hand4whole_dir):
        print("Hand4Whole directory not found. Exiting.")
        sys.exit(1)
        
    os.chdir(hand4whole_dir)
    if not execute_command(f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python run_hand4whole.py --gpu {args.gpu_ids} --root_path "{root_path}" --output_dir {os.path.join(base_output_dir, "smplx_init")}',
                          "Hand4Whole failed"):
        sys.exit(1)
    os.chdir('../../')
    
    # MMPOSE (get 2D whole-body keypoints)
    mmpose_dir = './mmpose'
    if not validate_path(mmpose_dir):
        print("mmpose directory not found. Exiting.")
        sys.exit(1)
        
    os.chdir(mmpose_dir)
    if not execute_command(f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python run_mmpose.py --root_path {root_path}', 'Failed to run mmpose'):
        print("mmpose failed")
        sys.exit(1)
    os.chdir('..')
    
    # fit SMPLX
    main_dir = '../main'
    if not validate_path(main_dir):
        print("main directory not found. Exiting.")
        sys.exit(1)
        
    os.chdir(main_dir)
    if not execute_command(f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python fit.py --subject_id "{subject_id}" --output_dir {base_output_dir} --gpu {args.gpu_ids}',
                          "fitting failed"):
        sys.exit(1)
    
    #unwrap textures of FLAME
    print(f"Unwrapping textures for {subject_id}, output to {base_output_dir}")
    if not execute_command(f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python unwrap.py --subject_id "{subject_id}" --output_dir {os.path.join(base_output_dir, "smplx_optimized")} --gpu {args.gpu_ids}',
                          "unwrapping failed"):
        sys.exit(1)
    
    # smooth SMPLX
    os.chdir('../tools')
    cmd =  f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python smooth_smplx_params.py --root_path {root_path}'
    print(cmd)
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when smoothing smplx parameters. terminate the script.')
        sys.exit()    
        
    os.chdir('../tools')
    # get foreground masks
    os.chdir('./segment-anything/')
    cmd = f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python run_sam.py --root_path {root_path}'
    result = os.system(cmd)
    os.chdir('..')
    if(result !=0):
        print("something bad happened when running SAM.")


    # get background point cloud
    os.chdir('./Depth-Anything-V2')
    cmd = f'CUDA_VISIBLE_DEVICES={args.gpu_ids} python run_depth_anything.py --root_path {root_path}'
    result = os.system(cmd)
    if (result != 0):
        print('something bad happened when running depth anything.')
        sys.exit()
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--use_colmap', dest='use_colmap', action='store_true')
    parser.add_argument('--gpu', type=str, dest='gpu_ids', required=True, help="GPU ID to use for processing")
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

def validate_path(path, create=False):
    if not osp.exists(path):
        print(f"Path does not exist: {path}")
        if create:
            try:
                os.makedirs(path, exist_ok=True)
                return True
            except Exception as e:
                print(f"Failed to create directory: {e}")
                return False
        return False
    return True

def execute_command(cmd, error_message):
    print(f"Executing: {cmd}")
    result = os.system(cmd)
    if result != 0:
        print(f"ERROR: {error_message}. Exit code: {result}")
        return False
    return True    

if __name__ == "__main__":
    main()