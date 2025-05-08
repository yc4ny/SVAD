from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import json
import argparse

def get_bbox(kpt_img, kpt_valid, extend_ratio=1.2):
    x_img, y_img = kpt_img[:,0], kpt_img[:,1]
    x_img = x_img[kpt_valid==1]; y_img = y_img[kpt_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5 * width * extend_ratio
    xmax = x_center + 0.5 * width * extend_ratio
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5 * height * extend_ratio
    ymax = y_center + 0.5 * height * extend_ratio

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

# Helper function to get frame filename with zero padding
def get_frame_filename(frame_idx):
    # Format with 4 digits padding (0000, 0001, etc.)
    return f"{frame_idx:04d}.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
out_path = osp.join(root_path, 'masks')
os.makedirs(out_path, exist_ok=True)

# Check for model file
ckpt_path = './sam_vit_h_4b8939.pth'
if not osp.exists(ckpt_path):
    print(f"WARNING: SAM checkpoint not found at {ckpt_path}")
    print("Please download the SAM model from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    exit(1)

# load SAM 
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=ckpt_path).cuda()
predictor = SamPredictor(sam)

# Check what frames exist in the frames directory
frames_dir = osp.join(root_path, 'frames')
print(f"Checking frames in: {frames_dir}")
frame_files = sorted(glob(osp.join(frames_dir, '*.png')))
if not frame_files:
    print(f"ERROR: No PNG files found in frames directory: {frames_dir}")
    exit(1)

print(f"Found {len(frame_files)} frames")
print(f"First few files: {[osp.basename(f) for f in frame_files[:5]]}")

# Extract frame indices from filenames
frame_idx_list = []
for img_path in frame_files:
    basename = osp.basename(img_path)
    # This handles both "0.png" and "0000.png" formats
    frame_idx = int(basename.split('.')[0])
    frame_idx_list.append(frame_idx)
frame_idx_list.sort()

print(f"Processing {len(frame_idx_list)} frames")

# Get image dimensions for video
first_img = cv2.imread(frame_files[0])
if first_img is None:
    print(f"ERROR: Could not read image: {frame_files[0]}")
    exit(1)
img_height, img_width = first_img.shape[:2]

# Set up video writer
video_save = cv2.VideoWriter(osp.join(root_path, 'masks.mp4'), 
                           cv2.VideoWriter_fourcc(*'mp4v'), 
                           30, 
                           (img_width*2, img_height))

# Process each frame
for frame_idx in tqdm(frame_idx_list):
    try:
        # Get padded frame filename
        frame_filename = get_frame_filename(frame_idx)
        
        # load image (first try padded format, then fallback to non-padded)
        img_path = osp.join(frames_dir, frame_filename)
        if not osp.exists(img_path):
            img_path = osp.join(frames_dir, f"{frame_idx}.png")
            
        if not osp.exists(img_path):
            print(f"WARNING: Frame file not found: {frame_filename}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: Could not read image: {img_path}")
            continue
        
        # load keypoints (first try padded format, then fallback to non-padded)
        kpt_path = osp.join(root_path, 'keypoints_whole_body', frame_filename)
        if not osp.exists(kpt_path):
            kpt_path = osp.join(root_path, 'keypoints_whole_body', f"{frame_idx}.png")
            
        if not osp.exists(kpt_path):
            kpt_path = osp.join(root_path, 'keypoints_whole_body', f"{frame_idx:04d}.json")
            
        if not osp.exists(kpt_path):
            kpt_path = osp.join(root_path, 'keypoints_whole_body', f"{frame_idx}.json")
            
        if not osp.exists(kpt_path):
            print(f"WARNING: Keypoints file not found for frame {frame_idx}")
            continue
            
        with open(kpt_path) as f:
            kpt = np.array(json.load(f), dtype=np.float32)
        
        # Filter keypoints with confidence score > 0.5
        valid_kpts = kpt[kpt[:,2] > 0.5,:2]
        if len(valid_kpts) == 0:
            print(f"WARNING: No valid keypoints found for frame {frame_idx}")
            continue
            
        bbox = get_bbox(valid_kpts, np.ones_like(valid_kpts[:,0]))
        bbox[2:] += bbox[:2]  # xywh -> xyxy

        # use keypoints as prompts
        img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_input)
        
        # First prediction
        masks, scores, logits = predictor.predict(
            point_coords=valid_kpts, 
            point_labels=np.ones_like(valid_kpts[:,0]), 
            box=bbox[None,:], 
            multimask_output=False
        )
        
        # Use best mask for second prediction
        mask_input = logits[np.argmax(scores), :, :]
        masks, _, _ = predictor.predict(
            point_coords=valid_kpts, 
            point_labels=np.ones_like(valid_kpts[:,0]), 
            box=bbox[None,:], 
            multimask_output=False, 
            mask_input=mask_input[None]
        )
        
        mask = masks.sum(0) > 0

        # save mask with proper zero-padded filename
        output_mask_path = osp.join(out_path, f"{frame_idx:04d}.png")
        cv2.imwrite(output_mask_path, mask * 255)
        
        # Create visualization and add to video
        img_masked = img.copy()
        img_masked[~mask] = 0
        frame = np.concatenate((img, img_masked), 1)
        frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), 
                          cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
        video_save.write(frame.astype(np.uint8))
        
    except Exception as e:
        print(f"ERROR processing frame {frame_idx}: {e}")
        continue

# Release video writer
video_save.release()
print(f"Processing complete. Results saved to {out_path}")
print(f"Mask video saved to {osp.join(root_path, 'masks.mp4')}")