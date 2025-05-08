#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import torch
import pickle
import numpy as np
from tqdm.rich import tqdm

from engines import LMDBEngine, CoreEngine

class Tracker:
    def __init__(self, focal_length, device='cuda'):
        self._device = device
        self.tracker = CoreEngine(focal_length=focal_length, device=device)

    def track_video(self, video_path, dir_path=None, no_vis=False, data_name=None):
        # build name
        data_name = os.path.basename(video_path).split('.')[0] if data_name is None else data_name
        output_path = os.path.join(f'outputs/{dir_path}', data_name) if dir_path else f'outputs/{data_name}'
        print('Building video data...')
        fps = self.tracker.build_video(video_path, output_path, matting=True, background=0.0)
        if fps is None:
            return
        print('Building video data done!')
        lmdb_engine = LMDBEngine(os.path.join(output_path, 'img_lmdb'), write=False)
        print('Track with flame/bbox...')
        base_results = self.tracker.track_base(lmdb_engine, output_path)
        print('Track with flame/bbox done!')
        print('Track optim...')
        optim_results = self.tracker.track_optim(base_results, output_path, lmdb_engine, share_id=True)
        print('Track optim done!')
        if not os.path.exists(os.path.join(output_path, 'smoothed.pkl')):
            optim_results = run_smoothing(optim_results, output_path)
        if not no_vis:
            self.run_visualization(optim_results, lmdb_engine, output_path, fps=fps)
        lmdb_engine.close()

    def run_visualization(self, data_result, lmdb_engine, output_path, fps=25.0):
        import torchvision
        from engines import FLAMEModel, RenderMesh
        self.verts_scale = self.tracker.calibration_results['verts_scale']
        self.focal_length = self.tracker.calibration_results['focal_length']
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=self.verts_scale).to(self._device)
        print('Visualize results...')
        mesh_render = RenderMesh(
            512, faces=self.flame_model.get_faces().cpu().numpy(), device=self._device
        )
        frames = list(data_result.keys())
        frames = sorted(frames, key=lambda x: int(x.split('_')[-1]))[:250]
        vis_images = []
        for frame in tqdm(frames, ncols=80, colour='#95bb72'):
            vertices, _ = self.flame_model(
                shape_params=torch.tensor(data_result[frame]['shapecode'], device=self._device)[None],
                expression_params=torch.tensor(data_result[frame]['expcode'], device=self._device)[None],
                pose_params=torch.tensor(data_result[frame]['posecode'], device=self._device)[None].float(),
                eye_pose_params=torch.tensor(data_result[frame]['eyecode'], device=self._device)[None].float()
            )
            images, alpha_images = mesh_render(
                vertices, focal_length=self.focal_length,
                transform_matrix=torch.tensor(data_result[frame]['transform_matrix'], device=self._device)[None],
            )
            alpha_images = alpha_images[0].expand(3, -1, -1)
            vis_image_0 = lmdb_engine[frame].to(self._device).float()
            vis_image_1 = vis_image_0.clone()
            vis_image_1[alpha_images>0.5] *= 0.3
            vis_image_1[alpha_images>0.5] += (images[0, alpha_images>0.5] * 0.7)
            vis_image_1 = vis_image_1.cpu().to(torch.uint8)
            vis_image = torchvision.utils.make_grid([vis_image_0.cpu(), vis_image_1.cpu(), images[0].cpu()], nrow=3, padding=0)[None]
            vis_images.append(vis_image)
        vis_images = torch.cat(vis_images, dim=0).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            os.path.join(output_path, 'tracked.mp4'), vis_images, fps=fps
        )
        print('Visualize results done!')


def run_smoothing(lightning_result, output_path):
    from copy import deepcopy
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    def smooth_params(data, alpha=0.7):
        smoothed_data = [data[0]]  # Initialize the smoothed data with the first value of the input data
        for i in range(1, len(data)):
            smoothed_value = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
            smoothed_data.append(smoothed_value)
        return smoothed_data
    smoothed_results = {}
    expression, pose, eyecode, rotates, translates = [], [], [], [], []
    frames = list(lightning_result.keys())
    frames = sorted(frames, key=lambda x: int(x.split('_')[-1]))
    for frame_name in frames:
        smoothed_results[frame_name] = deepcopy(lightning_result[frame_name])
        transform_matrix = smoothed_results[frame_name]['transform_matrix']
        rotates.append(matrix_to_rotation_6d(torch.tensor(transform_matrix[:3, :3])).numpy())
        translates.append(transform_matrix[:3, 3])
        eyecode.append(smoothed_results[frame_name]['eyecode'])
        # pose.append(smoothed_results[frame_name]['emoca_pose'])
        # expression.append(smoothed_results[frame_name]['emoca_expression'])
    # pose = smooth_params(np.stack(pose), alpha=0.95)
    # expression = smooth_params(np.stack(expression), alpha=0.95)
    print('Smoothing...')
    rotates = smooth_params(np.stack(rotates), alpha=0.6)
    translates = smooth_params(np.stack(translates), alpha=0.6)
    eyecode = smooth_params(np.stack(eyecode), alpha=0.7)
    print('Smoothing done!')
    # rotates = kalman_smooth_params(np.stack(rotates))
    # translates = kalman_smooth_params(np.stack(translates))
    for fidx, frame_name in enumerate(frames):
        # smoothed_results[frame_name]['emoca_pose'] = pose[fidx]
        # smoothed_results[frame_name]['emoca_expression'] = expression[fidx]
        rotation = rotation_6d_to_matrix(torch.tensor(rotates[fidx])).numpy()
        affine_matrix = np.concatenate([rotation, translates[fidx][:, None]], axis=-1)
        smoothed_results[frame_name]['transform_matrix'] = affine_matrix
        smoothed_results[frame_name]['eyecode'] = eyecode[fidx]
    with open(os.path.join(output_path, 'smoothed.pkl'), 'wb') as f:
        pickle.dump(smoothed_results, f)
    return smoothed_results


def list_all_files(dir_path):
    pair = os.walk(dir_path)
    result = []
    for path, dirs, files in pair:
        if len(files):
            for file_name in files:
                result.append(os.path.join(path, file_name))
    return result


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-v', required=True, type=str)
    parser.add_argument('--outdir_path', '-d', default='', type=str)
    parser.add_argument('--no_vis', action='store_true')
    args = parser.parse_args()
    tracker = Tracker(focal_length=12.0, device='cuda')
    if not os.path.isdir(args.video_path):
        tracker.track_video(args.video_path, dir_path=args.outdir_path, no_vis=args.no_vis)
    else:
        all_videos = list_all_files(args.video_path)
        all_videos = [v for v in all_videos if v.endswith('.mp4')]
        all_videos = sorted(all_videos)
        # all_videos = [v for i, v in enumerate(all_videos) if i % 2 == args.split_id]
        for vidx, video_path in enumerate(all_videos):
            print('Processing {}/{}......'.format(vidx+1, len(all_videos)))
            print(video_path)
            video_name = '_'.join([i for i in video_path.replace(args.video_path, '').split('/') if len(i)])
            tracker.track_video(video_path, dir_path=args.outdir_path, no_vis=args.no_vis, data_name=video_name)
