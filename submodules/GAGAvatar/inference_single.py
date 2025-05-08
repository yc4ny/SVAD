#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import argparse
import lightning
import numpy as np
import torchvision
from tqdm.rich import tqdm

from core.data import DriverData
from core.models import build_model
from core.libs.utils import ConfigDict
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine

def inference(image_path, driver_path, resume_path, output_dir, force_retrack=False, device='cuda'):
    lightning.fabric.seed_everything(42)
    driver_path = driver_path[:-1] if driver_path.endswith('/') else driver_path
    driver_name = os.path.basename(driver_path).split('.')[0]
    
    # Load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0])
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    
    print(str(meta_cfg))
    track_engine = TrackEngine(focal_length=12.0, device=device)
    
    # Build input data - ALWAYS force retrack the input image to ensure correct identity
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=True)
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    
    # Build driver data
    if os.path.isdir(driver_path):
        driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
        driver_dataset = DriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    else:
        driver_name = os.path.basename(driver_path).split('.')[0]
        driver_data = get_tracked_results(driver_path, track_engine, force_retrack=force_retrack)
        if driver_data is None:
            print(f'Finish inference, no face in driver: {image_path}.')
            return
        driver_dataset = DriverData({driver_name: driver_data}, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)

    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    
    # Run inference process
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Generating frames with identity from: {image_path}')
    print(f'Using driver expressions from: {driver_path}')
    print(f'Saving results to: {output_dir}')
    
    for idx, batch in enumerate(tqdm(driver_dataloader)):
        render_results = model.forward_expression(batch)
        # We only want the generated image with the input identity and driver expression
        pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
        
        # Save only the generated image (not a grid)
        dump_path = os.path.join(output_dir, f'{idx+1:04d}.png')
        torchvision.utils.save_image(pred_sr_rgb[0], dump_path)
        print(f'Saved frame {idx+1}: {dump_path}')

def get_tracked_results(image_path, track_engine, force_retrack=False):
    if not is_image(image_path):
        print(f'Please input a image path, got {image_path}.')
        return None
        
    # Create subject-specific tracking cache
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path).split('.')[0]
    subject_name = os.path.basename(os.path.dirname(image_path)) if image_dir else image_name
    
    tracked_folder = f'render_results/tracked/{subject_name}'
    tracked_pt_path = f'{tracked_folder}/tracked.pt'
    
    # Create directory if it doesn't exist
    if not os.path.exists(tracked_folder):
        os.makedirs(tracked_folder, exist_ok=True)
        
    # Initialize empty cache if it doesn't exist
    if not os.path.exists(tracked_pt_path):
        torch.save({}, tracked_pt_path)
        
    tracked_data = torch.load(tracked_pt_path)
    image_base = os.path.basename(image_path)
    
    if image_base in tracked_data and not force_retrack:
        print(f'Load tracking result from cache: {tracked_pt_path}.')
    else:
        print(f'Tracking {image_path}...')
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        feature_data = track_engine.track_image([image], [image_path])
        if feature_data is not None:
            feature_data = feature_data[image_path]
            # Save visualization of tracked features
            vis_path = f'{tracked_folder}/{image_base.split(".")[0]}_tracked.jpg'
            torchvision.utils.save_image(
                torch.tensor(feature_data['vis_image']), 
                vis_path
            )
            print(f'Saved tracked visualization: {vis_path}')
        else:
            print(f'No face detected in {image_path}.')
            return None
        
        # Update the tracked data for this image
        tracked_data[image_base] = feature_data
        torch.save(tracked_data, tracked_pt_path)
        
    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data


def is_image(image_path):
    extension_name = image_path.split('.')[-1].lower()
    return extension_name in ['jpg', 'png', 'jpeg']


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--driver_path', '-d', required=True, type=str)
    parser.add_argument('--output', '-o', required=True, type=str, help='Output directory to save the generated images')
    parser.add_argument('--force_retrack', '-f', action='store_true')
    parser.add_argument('--resume_path', '-r', default='./assets/GAGAvatar.pt', type=str)
    parser.add_argument('--gpu', type=str, help='GPU ID to use')
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU: {args.gpu}")
    
    # launch
    torch.set_float32_matmul_precision('high')
    inference(args.image_path, args.driver_path, args.resume_path, args.output, args.force_retrack)