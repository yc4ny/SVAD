import numpy as np
import torch
import torch.utils.data
import cv2
import os.path as osp
from glob import glob
from config import cfg
from utils.smpl_x import smpl_x
from utils.flame import flame
from utils.preprocessing import load_img, get_bbox 
from utils.transforms import transform_joint_to_other_db
from pytorch3d.transforms import quaternion_to_matrix
import json

class NeuMan(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.data_split = data_split
        self.root_path = osp.join('..', 'data', 'NeuMan', 'data', cfg.subject_id)
        self.transform = transform
        self.cam_params, self.img_paths, self.mask_paths, self.kpts, self.smplx_params, self.scene, self.frame_idx_list = self.load_data()
        self.load_id_info()
        self.cam_dist = self.get_cam_dist()

    def load_data(self):

        # read split file
        if cfg.fit_pose_to_test:
            split_path = osp.join(self.root_path, 'test_split.txt')
        else:
            split_path = osp.join(self.root_path, 'train_split.txt')
        with open(split_path) as f:
            frame_idx_list = [int(x[:-5]) for x in f.readlines()]

        # load cameras
        cam_params = {}
        with open(osp.join(self.root_path, 'sparse', 'cameras.txt')) as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            splitted = line.split()
            _, _, width, height, focal_x, focal_y, princpt_x, princpt_y = splitted
        focal = np.array((float(focal_x), float(focal_y)), dtype=np.float32) # shared across all frames
        princpt = np.array((float(princpt_x), float(princpt_y)), dtype=np.float32) # shared across all frames
        with open(osp.join(self.root_path, 'sparse', 'images.txt')) as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            if 'png' not in line:
                continue
            splitted = line.split()
            frame_idx = int(splitted[-1][:-4])
            qw, qx, qy, qz = float(splitted[1]), float(splitted[2]), float(splitted[3]), float(splitted[4])
            tx, ty, tz = float(splitted[5]), float(splitted[6]), float(splitted[7])
            R = quaternion_to_matrix(torch.FloatTensor([qw, qx, qy, qz])).numpy()
            t = np.array([tx, ty, tz], dtype=np.float32)
            cam_params[frame_idx] = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}

        # load image paths
        img_paths = {}
        img_path_list = glob(osp.join(self.root_path, 'images', '*.png'))
        for img_path in img_path_list:
            frame_idx = int(img_path.split('/')[-1][:-4])
            img_paths[frame_idx] = img_path
 
        # load mask paths
        mask_paths = {}
        mask_path_list = glob(osp.join(self.root_path, 'masks', '*.png'))
        for mask_path in mask_path_list:
            frame_idx = int(mask_path.split('/')[-1][:-4])
            mask_paths[frame_idx] = mask_path

        # load keypoints
        kpts = {}
        kpt_path_list = glob(osp.join(self.root_path, 'keypoints_whole_body', '*.json'))
        for kpt_path in kpt_path_list:
            frame_idx = int(kpt_path.split('/')[-1][:-5])
            with open(kpt_path) as f:
                kpts[frame_idx] = np.array(json.load(f), dtype=np.float32)

        # load smplx parameters
        smplx_params = {}
        smplx_param_path_list = glob(osp.join(self.root_path, 'smplx_optimized', 'smplx_params', '*.json'))
        for smplx_param_path in smplx_param_path_list:
            file_name = smplx_param_path.split('/')[-1]
            frame_idx = int(file_name[:-5])
            with open(smplx_param_path) as f:
                smplx_params[frame_idx] = {k: torch.FloatTensor(v) for k,v in json.load(f).items()}

        # load point cloud of scene
        scene = []
        with open(osp.join(self.root_path, 'sparse', 'points3D.txt')) as f:
            lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            splitted = line.split()
            xyz = torch.FloatTensor([float(splitted[1]), float(splitted[2]), float(splitted[3])])
            rgb = torch.FloatTensor([float(splitted[4]), float(splitted[5]), float(splitted[6])]) / 255
            scene.append(torch.cat((xyz, rgb)))
        scene = torch.stack(scene)
        is_valid = scene[:,2] < torch.quantile(scene[:,2], 0.95) # remove outliers
        scene = scene[is_valid,:]
        
        if self.data_split == 'train':
            frame_idx_list *= 100
        return cam_params, img_paths, mask_paths, kpts, smplx_params, scene, frame_idx_list
    
    def load_id_info(self):
        with open(osp.join(self.root_path, 'smplx_optimized', 'shape_param.json')) as f:
            shape_param = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'face_offset.json')) as f:
            face_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'joint_offset.json')) as f:
            joint_offset = torch.FloatTensor(json.load(f))
        with open(osp.join(self.root_path, 'smplx_optimized', 'locator_offset.json')) as f:
            locator_offset = torch.FloatTensor(json.load(f))
        smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

        texture_path = osp.join(self.root_path, 'smplx_optimized', 'face_texture.png')
        texture = torch.FloatTensor(cv2.imread(texture_path)[:,:,::-1].copy().transpose(2,0,1))/255
        texture_mask_path = osp.join(self.root_path, 'smplx_optimized', 'face_texture_mask.png')
        texture_mask = torch.FloatTensor(cv2.imread(texture_mask_path).transpose(2,0,1))/255
        flame.set_texture(texture, texture_mask)

    def __len__(self):
        return len(self.frame_idx_list)
    
    def __getitem__(self, idx):
        frame_idx = self.frame_idx_list[idx]

        # load image
        img = load_img(self.img_paths[frame_idx])
        img = self.transform(img.astype(np.float32))/255.

        # load mask
        mask = cv2.imread(self.mask_paths[frame_idx])[:,:,0,None] / 255.
        mask = self.transform((mask > 0.5).astype(np.float32))

        # get bbox from 2D keypoints
        joint_img = self.kpts[frame_idx][:,:2]
        joint_valid = (self.kpts[frame_idx][:,2:] > 0.5).astype(np.float32)
        bbox = get_bbox(joint_img, joint_valid[:,0])

        data = {'img': img, 'mask': mask, 'bbox': bbox, 'cam_param': self.cam_params[frame_idx], 'frame_idx': frame_idx}
        return data

    # get camera distribution of a scene
    def get_cam_dist(self):
        cam_pos_list = []
        for frame_idx in self.cam_params.keys():
            R, t = self.cam_params[frame_idx]['R'], self.cam_params[frame_idx]['t']
            cam_pos = np.dot(R.transpose(1,0), -t.reshape(3)).reshape(3)
            cam_pos_list.append(cam_pos)
        cam_pos_list = np.stack(cam_pos_list)
        
        cam_pos_mean = np.mean(cam_pos_list, 0)
        dist_max = np.max(np.sqrt(np.sum((cam_pos_list - cam_pos_mean[None,:])**2,1)))
        translate = torch.from_numpy(-cam_pos_mean).float()
        radius = torch.ones((1)).float() * float(dist_max) * 1.1
        return {'translate': translate, 'radius': radius}
