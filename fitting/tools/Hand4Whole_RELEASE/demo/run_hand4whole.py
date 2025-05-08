import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh
import json
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from glob import glob
from tqdm import tqdm

def get_one_box(det_output):
    max_score = 0
    max_bbox = None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = score

    return max_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', required=True)
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--output_dir', type=str, dest='output_dir', help="Custom output directory")
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# Get the zero-padded frame filename based on frame index
def get_frame_filename(frame_idx):
    # Format with 4 digits padding (0000, 0001, etc.)
    return f"{frame_idx:04d}.png"

args = parse_args()
# Set GPU explicitly
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
print(f"Using GPU: {args.gpu_ids}")

cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
root_path = args.root_path

# Set output directory
if args.output_dir:
    save_path = args.output_dir
else:
    save_path = osp.join(root_path, 'smplx_init')

# Create output directory
os.makedirs(save_path, exist_ok=True)

# snapshot load
model_path = '../snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()

# Check what files actually exist in the frames directory
frames_dir = osp.join(root_path, 'frames')
print(f"Checking frames directory: {frames_dir}")
if not osp.exists(frames_dir):
    print(f"ERROR: Frames directory does not exist: {frames_dir}")
    sys.exit(1)

# List all image files and print some examples for debugging
all_files = glob(osp.join(frames_dir, '*.png'))
print(f"Found {len(all_files)} PNG files in frames directory")

# Modify to match 4-digit format for frame files (0000.png, 0001.png, etc.)
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
if not img_path_list:
    print(f"ERROR: No PNG files found in frames directory: {frames_dir}")
    sys.exit(1)

# Get the first image to determine dimensions
first_img_path = img_path_list[0]
print(f"Reading first image: {first_img_path}")
first_img = cv2.imread(first_img_path)
if first_img is None:
    print(f"ERROR: Could not read image: {first_img_path}")
    sys.exit(1)

img_height, img_width = first_img.shape[:2]
print(f"Image dimensions: {img_width}x{img_height}")

# Extract frame indices from filenames, which may be in format "0000.png"
frame_idx_list = []
for img_path in img_path_list:
    base_name = osp.basename(img_path)
    frame_idx = int(base_name.split('.')[0])  # This will work for both "0.png" and "0000.png"
    frame_idx_list.append(frame_idx)
frame_idx_list.sort()

print(f"Processing {len(frame_idx_list)} frames")
bbox = None
det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
det_transform = T.Compose([T.ToTensor()])

for frame_idx in tqdm(frame_idx_list):
    # Use zero-padded frame filename (0000.png, 0001.png, etc.)
    frame_filename = get_frame_filename(frame_idx)
    img_path = osp.join(root_path, 'frames', frame_filename)
    
    if not osp.exists(img_path):
        print(f"WARNING: Frame file does not exist: {img_path}")
        continue
        
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"WARNING: Could not read image: {img_path}")
        continue
        
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img_height, original_img_width = original_img.shape[:2]

    # prepare bbox
    det_input = det_transform(original_img).cuda()
    det_output = det_model([det_input])[0]
    bbox = get_one_box(det_output) # xyxy
    if bbox is None:
        print(f"WARNING: No person detected in frame {frame_idx}")
        continue
    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] # xywh
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

    # render mesh
    vis_img = original_img[:,:,::-1].copy()
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
    frame = np.concatenate((vis_img, rendered_img),1)
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)

    # save SMPL-X parameters
    root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
    body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
    lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
    rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
    jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
    shape = out['smplx_shape'].detach().cpu().numpy()[0]
    expr = out['smplx_expr'].detach().cpu().numpy()[0] 
    
    # Save with the original frame index
    with open(osp.join(save_path, f"{frame_idx:04d}.json"), 'w') as f:
        json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                'body_pose': body_pose.reshape(-1,3).tolist(), \
                'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                'leye_pose': [0,0,0],\
                'reye_pose': [0,0,0],\
                'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                'shape': shape.reshape(-1).tolist(), \
                'expr': expr.reshape(-1).tolist()}, f)


# Create symlinks or copy files back to root_path if using custom output directory
if args.output_dir and args.output_dir != osp.join(root_path, 'smplx_init'):
    root_init_dir = osp.join(root_path, 'smplx_init')
    os.makedirs(root_init_dir, exist_ok=True)
    
    # Copy all the files
    for json_file in glob(osp.join(save_path, '*.json')):
        try:
            target_file = osp.join(root_init_dir, osp.basename(json_file))
            import shutil
            shutil.copy2(json_file, target_file)
        except Exception as e:
            print(f"Error copying {json_file}: {e}")
    
    print(f"Copied SMPLX param files to {root_init_dir}")

print(f"Processing complete. Results saved to {save_path}")