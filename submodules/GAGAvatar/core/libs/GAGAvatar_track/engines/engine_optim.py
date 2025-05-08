#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import math
import torch
import torchvision
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .flame_model import FLAMEModel, RenderMesh
from .vgghead_detector import reproject_vertices

class OptimEngine:
    def __init__(self, calibration_results, device='cuda'):
        self._device = device
        self.image_size = 512.0
        self.verts_scale = calibration_results['verts_scale']
        self.focal_length = torch.tensor([calibration_results['focal_length']], device=device)
        # build flame
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=self.verts_scale).to(device)

    def _build_cameras_kwargs(self, batch_size):
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self._device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self._device).float(), 
            'focal_length': self.focal_length, 
            'image_size': screen_size, 'device': self._device,
        }
        return cameras_kwargs

    def lightning_optimize(self, track_frames, batch_base, batch_frames=None, steps=300, share_id=False):
        batch_size = len(track_frames)
        cameras_kwargs = self._build_cameras_kwargs(batch_size)
        cameras = PerspectiveCameras(**cameras_kwargs)
        batch_emica, batch_vgg = batch_base['emica_results'], batch_base['vgg_results']
        # vgg head params
        vgg_vertices, vgg_lmks2d70 = reproject_vertices(self.flame_model, batch_vgg)
        gt_lmks2d70 = batch_vgg['lmks_2d70']
        # emica head params
        emica_shapecode = batch_emica['shapecode'].mean(dim=0, keepdim=True).expand(batch_size, -1) if share_id else batch_emica['shapecode']
        emica_expcode = batch_emica['expcode']
        emica_posecode = torch.cat([batch_emica['jawpose'].new_zeros(batch_emica['jawpose'].shape), batch_emica['jawpose']], dim=1)
        emica_vertices, emica_lmks = self.flame_model(
            emica_shapecode, emica_expcode, emica_posecode
        )
        base_transform_p3d = self.transform_emoca_to_p3d(
            batch_base['emica_results']['globalpose'], 
            emica_vertices, vgg_vertices, self.image_size
        )
        ori_rotation = matrix_to_rotation_6d(base_transform_p3d[:, :3, :3])
        base_transform_p3d[:, :3, 3] = base_transform_p3d[:, :3, 3] * self.focal_length / 13.0 * self.verts_scale / 5.0
        rotation = torch.nn.Parameter(ori_rotation, requires_grad=True)
        translation = torch.nn.Parameter(base_transform_p3d[:, :, 3], requires_grad=True)
        # flame params
        eye_pose_code = torch.nn.Parameter(emica_posecode.new_zeros(emica_posecode.shape), requires_grad=True)
        optimizer = torch.optim.Adam([
            {'params': [eye_pose_code], 'lr': 0.001},
            {'params': [rotation], 'lr': 0.0005}, {'params': [translation], 'lr': 0.1},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        # run
        lmks_distance = (gt_lmks2d70[:, 17:68] - vgg_lmks2d70[:, 17:68]).norm(dim=-1).mean(dim=-1)
        lmks_mask = lmks_distance < 8.96
        # print(f"Start optimize: {lmks_mask.sum()} / {lmks_mask.shape[0]}.")
        for idx in range(steps):
            pred_vertices, pred_lmks = self.flame_model(
                shape_params=emica_shapecode, expression_params=emica_expcode, 
                pose_params=emica_posecode, eye_pose_params=eye_pose_code
            )
            project_vertices = cameras.transform_points_screen(
                pred_vertices, R=rotation_6d_to_matrix(rotation), T=translation
            )[..., :2]
            pred_lmks = cameras.transform_points_screen(
                pred_lmks, R=rotation_6d_to_matrix(rotation), T=translation
            )[..., :2]
            loss_vertices = mse_loss(project_vertices, vgg_vertices) * 25
            loss_lmks = mse_loss(pred_lmks[:, 17:68], gt_lmks2d70[:, 17:68], mask=lmks_mask) * 10
            loss_lmks1 = mse_loss(pred_lmks[:, :68], vgg_lmks2d70[:, :68], mask=lmks_mask) * 80
            loss_eyepose = mse_loss(pred_lmks[:, 68:], gt_lmks2d70[:, 68:], mask=lmks_mask) * 25
            base_loss = loss_vertices + loss_eyepose + loss_lmks + loss_lmks1
            optimizer.zero_grad()
            base_loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            
        optim_results = {}
        head_bbox = batch_base['bbox'].clamp(0.0, 1.0)
        transform_matrix = torch.cat([rotation_6d_to_matrix(rotation), translation[:, :, None]], dim=-1)
        for idx, name in enumerate(track_frames):
            optim_results[name] = {
                'bbox': head_bbox[idx].detach().float().cpu().numpy(),
                'shapecode': emica_shapecode[idx].detach().float().cpu().numpy(),
                'expcode': emica_expcode[idx].detach().float().cpu().numpy(),
                'posecode': emica_posecode[idx].detach().float().cpu().numpy(),
                'eyecode': eye_pose_code[idx].detach().float().cpu().numpy(),
                'transform_matrix': transform_matrix[idx].detach().float().cpu().numpy(),
            }
        if batch_frames is not None:
            with torch.no_grad():
                mesh_render = RenderMesh(
                    512, faces=self.flame_model.get_faces().cpu().numpy(), device=self._device
                )
                images, alpha_images = mesh_render(
                    pred_vertices[:batch_frames.shape[0]], focal_length=self.focal_length,
                    transform_matrix=transform_matrix[:batch_frames.shape[0]],
                )
                vis_images = []
                alpha_images = alpha_images.expand(-1, 3, -1, -1)
                for idx, frame in enumerate(batch_frames):
                    vis_i = frame.clone()
                    vis_i[alpha_images[idx]>0.5] *= 0.5
                    vis_i[alpha_images[idx]>0.5] += (images[idx, alpha_images[idx]>0.5] * 0.5)
                    bbox = head_bbox[idx].clone()
                    bbox[[0, 2]] *= vis_i.shape[-1]; bbox[[1, 3]] *= vis_i.shape[-2]
                    vis_i = torchvision.utils.draw_bounding_boxes(
                        vis_i.cpu().to(torch.uint8), bbox[None], width=3, colors='green'
                    )
                    # vis_i = torchvision.utils.draw_keypoints(vis_i, vgg_lmks2d70[idx:idx+1], colors="red", radius=1.5)
                    # vis_i = torchvision.utils.draw_keypoints(vis_i, vgg_vertices[idx:idx+1], colors="red", radius=0.5)
                    vis_i = torchvision.utils.draw_keypoints(vis_i, gt_lmks2d70[idx:idx+1, 17:], colors="blue", radius=1.5)
                    vis_i = torchvision.utils.draw_keypoints(vis_i, pred_lmks[idx:idx+1], colors="white", radius=1.5)
                    vis_images.append(vis_i.float().cpu()/255.0)
                visualization = torchvision.utils.make_grid(vis_images, nrow=4)
        else:
            visualization = None
        return optim_results, visualization

    @staticmethod
    def transform_emoca_to_p3d(emoca_base_rotation, pred_lmks, gt_lmks, image_size):
        device = emoca_base_rotation.device
        batch_size = emoca_base_rotation.shape[0]
        initial_trans = torch.tensor([[0, 0, 5000.0/image_size]]).to(device)
        emoca_base_rotation[:, 1] += math.pi
        emoca_base_rotation = emoca_base_rotation[:, [2, 1, 0]]
        emoca_base_rotation = batch_rodrigues(emoca_base_rotation)
        base_transform = torch.cat([
                transform_inv(emoca_base_rotation), 
                initial_trans.reshape(1, -1, 1).repeat(batch_size, 1, 1)
            ], dim=-1
        )
        base_transform_p3d = transform_opencv_to_p3d(base_transform)
        # find translate
        cameras = PerspectiveCameras(
            device=device,
            image_size=torch.tensor([[image_size, image_size]], device=device).repeat(batch_size, 1)
        )
        pred_lmks = cameras.transform_points_screen(
            pred_lmks.clone(),
            R=base_transform_p3d[:, :3, :3], T=base_transform_p3d[:, :3, 3], 
            principal_point=torch.zeros(batch_size, 2), focal_length=5000.0/image_size
        )[..., :2]
        trans_xy = (pred_lmks.mean(dim=1)[..., :2] - gt_lmks.mean(dim=1)[..., :2]) * 2 / image_size
        base_transform_p3d[:, :2, 3] = trans_xy
        return base_transform_p3d


def mse_loss(opt_lmks, target_lmks, mask=None):
    if mask is not None:
        diff = torch.nn.functional.mse_loss(opt_lmks[mask], target_lmks[mask]) / 512.0
    else:
        diff = torch.nn.functional.mse_loss(opt_lmks, target_lmks) / 512.0
    return diff


def intrinsic_opencv_to_p3d(focal_length, principal_point, image_size):
    half_size = image_size/2
    focal_length = focal_length / half_size
    principal_point = -(principal_point - half_size) / half_size
    return focal_length, principal_point


def transform_opencv_to_p3d(opencv_transform, verts_scale=1, type='w2c'):
    assert type in ['w2c', 'c2w']
    if opencv_transform.dim() == 3:
        return torch.stack([transform_opencv_to_p3d(t, verts_scale, type) for t in opencv_transform])
    if type == 'c2w':
        if opencv_transform.shape[-1] != opencv_transform.shape[-2]:
            new_transform = torch.eye(4).to(opencv_transform.device)
            new_transform[:3, :] = opencv_transform
            opencv_transform = new_transform
        opencv_transform = torch.linalg.inv(opencv_transform) # c2w to w2c
    rotation = opencv_transform[:3, :3]
    rotation = rotation.permute(1, 0)
    rotation[:, :2] *= -1
    if opencv_transform.shape[-1] == 4:
        translation = opencv_transform[:3, 3] * verts_scale
        translation[:2] *= -1
        rotation = torch.cat([rotation, translation.reshape(-1, 1)], dim=-1)
    return rotation


def transform_inv(transforms):
    if transforms.dim() == 3:
        return torch.stack([transform_opencv_to_p3d(t) for t in transforms])
    if transforms.shape[-1] != transforms.shape[-2]:
        new_transform = torch.eye(4)
        new_transform[:3, :] = transforms
        transforms = new_transform
    transforms = torch.linalg.inv(transforms)
    return transforms[:3]


def batch_rodrigues(rot_vecs,):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def expand_bbox(bbox, t_scale=1.0):
    xmin, ymin, xmax, ymax = bbox.split(1, dim=-1)
    cenx, ceny = ((xmin + xmax) / 2), ((ymin + ymax) / 2)
    target_w = (xmax - xmin) * t_scale
    target_h = (ymax - ymin) * t_scale
    # mask = target_w < target_h
    # target_w[mask] = target_w[mask] * 1.1
    # target_h[~mask] = target_h[~mask] * 1.1
    xmine, xmaxe = cenx - target_w / 2, cenx + target_w / 2
    ymine, ymaxe = ceny - target_h / 2, ceny + target_h / 2
    return torch.cat([xmine, ymine, xmaxe, ymaxe], dim=-1).clamp(0.0, 1.0)
