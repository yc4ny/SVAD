#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import time
import torch
import pickle
import shutil
import numpy as np
import torchvision

from engines import CoreEngine

class Tracker:
    def __init__(self, focal_length, device='cuda'):
        self._device = device
        self.tracker = CoreEngine(focal_length=focal_length, device=device)

    def track_image(self, image_paths, no_matting=False):
        # build name
        image_keys = [os.path.basename(image_path) for image_path in image_paths]
        if len(image_paths) == 1:
            output_path = 'outputs/{}'.format(os.path.basename(image_paths[0]))
        else:
            output_path = 'outputs/{}'.format(os.path.basename(os.path.dirname(image_paths[0])))
        if os.path.exists(output_path):
            print(f'Output path {output_path} exists, replace it.')
            shutil.rmtree(output_path) 
        os.makedirs(output_path)
        input_images = [
            torchvision.io.read_image(i, mode=torchvision.io.image.ImageReadMode.RGB).to(self._device).float()
            for i in image_paths
        ]
        print('Track image...')
        track_results = self.tracker.track_image(input_images, image_keys, if_matting=not no_matting)
        if track_results is not None:
            for key in track_results.keys():
                torchvision.utils.save_image(
                    [torch.tensor(track_results[key]['image']), torch.tensor(track_results[key]['vis_image'])],
                    os.path.join(output_path, key)
                )
                track_results[key].pop('vis_image')
            with open(os.path.join(output_path, 'optim.pkl'), 'wb') as f:
                pickle.dump(track_results, f)
            print('Track done: {}!'.format(output_path))


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--no_matting', action='store_true')
    args = parser.parse_args()
    
    tracker = Tracker(focal_length=12.0, device='cuda')
    if os.path.isdir(args.image_path):
        image_paths = os.listdir(args.image_path)
        image_paths = [image_path for image_path in image_paths if image_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
        image_paths = [os.path.join(args.image_path, image_path) for image_path in image_paths]
        tracker.track_image(image_paths, no_matting=args.no_matting)
    else:
        assert args.image_path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg'], 'Invalid image path!'
        tracker.track_image([args.image_path], no_matting=args.no_matting)
