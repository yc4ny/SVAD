#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from Jia Guo (insightface.ai).

import torch
import os, glob
import onnx2torch
import torchvision
import numpy as np

__all__ = ['InsightDetector']

class InsightDetector:
    def __init__(self, device='cuda'):
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join('assets/emica/ins_scrfd_10g_bnkps.onnx')
        assert os.path.exists(_model_path), f"Model not found: {_model_path}."
        self._device = device
        self.det_model = RetinaFace(_model_path, input_size=320, device=device)

    def get(self, image_tensor):
        bbox, kps = self.det_model.detect(image_tensor.to(self._device))
        if bbox.shape[0] == 0:
            return None, None, None
        faces = [dict(bbox=bbox[i, 0:4], kps=kps[i], det_score=bbox[i, 4]) for i in range(bbox.shape[0])]
        largest_face = max(faces, key=lambda x: (x['bbox'][3] - x['bbox'][1]) * (x['bbox'][2] - x['bbox'][0]))
        bbox, score, kps = largest_face['bbox'], largest_face['det_score'], largest_face['kps']
        return bbox, kps, score


class RetinaFace:
    def __init__(self, model_path, input_size=640, device='cuda'):
        self.device = device
        self.model = onnx2torch.convert(model_path)
        self.model.eval().to(device)
        # model
        self._num_anchors = 2
        self._center_cache = {}
        self._feat_stride_fpn = [8, 16, 32]
        self.input_size = (input_size, input_size)
        self.input_std, self.input_mean = 128.0, 127.5
        self.nms_thresh, self.det_thresh = 0.4, 0.5

    @torch.no_grad()
    def forward_model(self, img_tensor, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        img_tensor = (img_tensor[None] - self.input_mean) / self.input_std
        net_outs = self.model(img_tensor)
        input_height, input_width = img_tensor.shape[2:]
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds, kps_preds = net_outs[idx+3] * stride, net_outs[idx+6] * stride
            height, width = input_height // stride, input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self._center_cache:
                anchor_centers = self._center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = torch.tensor(anchor_centers).float().to(self.device)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                anchor_centers = torch.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1,2))
                if len(self._center_cache)<100:
                    self._center_cache[key] = anchor_centers
            pos_inds = torch.where(scores>=threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_scores, pos_bboxes, pos_kpss = scores[pos_inds], bboxes[pos_inds], kpss[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss_list.append(pos_kpss)
        scores_list = torch.cat(scores_list, dim=0).squeeze(1)
        bboxes_list = torch.cat(bboxes_list, dim=0)
        kpss_list = torch.cat(kpss_list, dim=0)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img):
        if img.shape[1] > img.shape[2]:
            img = torchvision.transforms.functional.pad(img, (0, 0, img.shape[1]-img.shape[2], 0))
        elif img.shape[2] > img.shape[1]:
            img = torchvision.transforms.functional.pad(img, (0, 0, 0, img.shape[2]-img.shape[1]))
        det_scale = float(self.input_size[0]) / img.shape[1]
        det_img = torchvision.transforms.functional.resize(img, self.input_size, antialias=True)
        scores, bboxes, kpss = self.forward_model(det_img, self.det_thresh)
        bboxes, kpss = bboxes / det_scale, kpss / det_scale
        keep = torchvision.ops.nms(bboxes, scores, self.nms_thresh)
        det = torch.cat((bboxes[keep], scores[keep].unsqueeze(1)), dim=1)
        kpss = kpss[keep]
        return det, kpss


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, axis=-1)

