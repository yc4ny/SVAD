import cv2
import numpy as np
import torch
import os.path as osp
from glob import glob
from pytorch3d.io import load_ply
import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
PerspectiveCameras,
RasterizationSettings,
MeshRasterizer)
import json
from tqdm import tqdm
import argparse

# Helper function to get frame filename with zero padding
def get_frame_filename(frame_idx):
    # Format with 4 digits padding (0000, 0001, etc.)
    return f"{frame_idx:04d}.png"

def render_depthmap(mesh, face, cam_param, render_shape):
    mesh = mesh.cuda()[None,:,:]
    face = face.cuda()[None,:,:]
    cam_param = {k: torch.FloatTensor(v).cuda()[None,:] for k,v in cam_param.items()}

    batch_size, vertex_num = mesh.shape[:2]
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'], 
                                principal_point=cam_param['princpt'], 
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=0, perspective_correct=True)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()

    # render
    with torch.no_grad():
        fragments = rasterizer(mesh)
    
    depthmap = fragments.zbuf.cpu().numpy()[0,:,:,0]
    return depthmap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
out_path = osp.join(root_path, 'depthmaps')
os.makedirs(out_path, exist_ok=True)

# Print directory info for debugging
print(f"Root path: {root_path}")
print(f"Output path: {out_path}")

# Check what frames exist
frames_dir = osp.join(root_path, 'frames')
print(f"Checking frames in: {frames_dir}")
frame_files = sorted(glob(osp.join(frames_dir, '*.png')))
if not frame_files:
    print(f"ERROR: No PNG files found in frames directory: {frames_dir}")
    exit(1)

print(f"Found {len(frame_files)} frames")
print(f"First few files: {[osp.basename(f) for f in frame_files[:5]]}")

# run DepthAnything-V2
cmd = f'python run.py --encoder vitl --img-path "{osp.join(root_path, "frames")}" --outdir "{out_path}" --pred-only --grayscale'
print(f"Running DepthAnything-V2: {cmd}")
result = os.system(cmd)
if result != 0:
    print(f"WARNING: DepthAnything-V2 command returned non-zero exit code: {result}")

# Check if depthmap files were created
depthmap_path_list = glob(osp.join(out_path, '*.png'))
if not depthmap_path_list:
    print(f"ERROR: No depthmap files found in output directory: {out_path}")
    exit(1)

print(f"Found {len(depthmap_path_list)} depthmap files")
print(f"First few files: {[osp.basename(f) for f in depthmap_path_list[:5]]}")

# Extract frame indices from filenames
frame_idx_list = []
for depthmap_path in depthmap_path_list:
    basename = osp.basename(depthmap_path)
    # This handles both "0.png" and "0000.png" formats
    frame_idx = int(basename.split('.')[0])
    frame_idx_list.append(frame_idx)
frame_idx_list.sort()

print(f"Processing {len(frame_idx_list)} frames")

# Get image dimensions for video
first_depthmap = cv2.imread(depthmap_path_list[0])
if first_depthmap is None:
    print(f"ERROR: Could not read depthmap: {depthmap_path_list[0]}")
    exit(1)
img_height, img_width = first_depthmap.shape[:2]

# Set up video writer
video_save = cv2.VideoWriter(osp.join(root_path, 'depthmaps.mp4'), 
                           cv2.VideoWriter_fourcc(*'mp4v'), 
                           30, 
                           (img_width*2, img_height))

# Initialize accumulation variables
depthmap_save = 0
color_save = 0
is_bkg_save = 0
last_valid_cam_param = None

# Process each frame
for frame_idx in tqdm(frame_idx_list):
    try:
        # Get padded frame filename
        frame_filename = get_frame_filename(frame_idx)
        
        # Load image (try both padded and non-padded formats)
        img_path = osp.join(frames_dir, frame_filename)
        if not osp.exists(img_path):
            img_path = osp.join(frames_dir, f"{frame_idx}.png")
            
        if not osp.exists(img_path):
            print(f"WARNING: Image file not found: {frame_filename}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Could not read image: {img_path}")
            continue
        
        img = img.astype(np.float32)

        # Load depthmap (try both padded and non-padded formats)
        depthmap_path = osp.join(out_path, frame_filename)
        if not osp.exists(depthmap_path):
            depthmap_path = osp.join(out_path, f"{frame_idx}.png")
            
        if not osp.exists(depthmap_path):
            print(f"WARNING: Depthmap not found for frame {frame_idx}")
            continue
            
        depthmap = cv2.imread(depthmap_path)
        if depthmap is None:
            print(f"WARNING: Could not read depthmap: {depthmap_path}")
            continue
        
        # Save video frame
        frame = np.concatenate((img, depthmap), 1)
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), 
                          cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video_save.write(frame.astype(np.uint8))
        
        # Load SMPLX mesh (try both padded and non-padded formats)
        smplx_mesh_path = osp.join(root_path, 'smplx_optimized', 'meshes_smoothed', f"{frame_idx:04d}_smplx.ply")
        if not osp.exists(smplx_mesh_path):
            smplx_mesh_path = osp.join(root_path, 'smplx_optimized', 'meshes_smoothed', f"{frame_idx}_smplx.ply")
            
        if not osp.exists(smplx_mesh_path):
            print(f"WARNING: SMPLX mesh not found for frame {frame_idx}")
            continue
            
        smplx_vert, smplx_face = load_ply(smplx_mesh_path)
        
        # Load camera parameters (try both padded and non-padded formats)
        cam_param_path = osp.join(root_path, 'cam_params', f"{frame_idx:04d}.json")
        if not osp.exists(cam_param_path):
            cam_param_path = osp.join(root_path, 'cam_params', f"{frame_idx}.json")
            
        if not osp.exists(cam_param_path):
            print(f"WARNING: Camera parameters not found for frame {frame_idx}")
            if last_valid_cam_param is None:
                continue
            else:
                print(f"Using last valid camera parameters")
                cam_param = last_valid_cam_param
        else:
            with open(cam_param_path) as f:
                cam_param = json.load(f)
                last_valid_cam_param = cam_param
        
        # Render depthmap from SMPLX mesh
        smplx_depthmap = render_depthmap(smplx_vert, smplx_face, cam_param, (img_height, img_width))
        smplx_is_fg = smplx_depthmap > 0

        # Normalize depthmap from DepthAnything-V2
        depthmap_values = 255 - depthmap[:,:,0]  # close points high values -> close points low values
        
        # Check if there are any foreground pixels
        if np.sum(smplx_is_fg) == 0:
            print(f"WARNING: No foreground pixels in SMPLX depthmap for frame {frame_idx}")
            continue
            
        scale = np.abs(depthmap_values[smplx_is_fg] - depthmap_values[smplx_is_fg].mean()).mean()
        scale_smplx = np.abs(smplx_depthmap[smplx_is_fg] - smplx_depthmap[smplx_is_fg].mean()).mean()
        
        if scale == 0 or np.isnan(scale) or np.isinf(scale):
            print(f"WARNING: Invalid scale value for frame {frame_idx}")
            continue
            
        depthmap_values = depthmap_values / scale * scale_smplx
        depthmap_values = depthmap_values - depthmap_values[smplx_is_fg].mean() + smplx_depthmap[smplx_is_fg].mean()

        # Load mask (try both padded and non-padded formats)
        mask_path = osp.join(root_path, 'masks', frame_filename)
        if not osp.exists(mask_path):
            mask_path = osp.join(root_path, 'masks', f"{frame_idx}.png")
            
        if not osp.exists(mask_path):
            print(f"WARNING: Mask not found for frame {frame_idx}")
            continue
            
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f"WARNING: Could not read mask: {mask_path}")
            continue
            
        mask = mask[:,:,0]
        is_bkg = mask < 0.5

        # Accumulate background points
        depthmap_save += depthmap_values * is_bkg
        color_save += img * is_bkg[:,:,None]
        is_bkg_save += is_bkg
        
    except Exception as e:
        print(f"ERROR processing frame {frame_idx}: {e}")
        continue

# Release video writer
video_save.release()

# Check if we have valid background points
if np.sum(is_bkg_save) == 0:
    print("ERROR: No valid background points found across all frames")
    exit(1)

print(f"Saving background point cloud...")

# Save background point cloud
depthmap_save = depthmap_save / (is_bkg_save + 1e-6)
color_save = color_save / (is_bkg_save[:,:,None] + 1e-6)

# Make sure we have a valid camera parameter for the final step
if last_valid_cam_param is None:
    print("ERROR: No valid camera parameters found in any frame")
    exit(1)

try:
    point_cloud_path = osp.join(root_path, 'bkg_point_cloud.txt')
    with open(point_cloud_path, 'w') as f:
        for i in tqdm(range(img_height)):
            for j in range(img_width):
                if is_bkg_save[i][j]:
                    x = (j - last_valid_cam_param['princpt'][0]) / last_valid_cam_param['focal'][0] * depthmap_save[i][j]
                    y = (i - last_valid_cam_param['princpt'][1]) / last_valid_cam_param['focal'][1] * depthmap_save[i][j]
                    z = depthmap_save[i][j]
                    rgb = color_save[i][j]
                    f.write(f"{x} {y} {z} {rgb[0]} {rgb[1]} {rgb[2]}\n")
                    
    print(f"Background point cloud saved to: {point_cloud_path}")
    print(f"Processing complete.")
except Exception as e:
    print(f"ERROR saving background point cloud: {e}")