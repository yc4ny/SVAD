#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import numpy as np
import face_alignment

class FANDetector:
    def __init__(self, device='cuda'):
        try:
            mode = face_alignment.LandmarksType._2D
        except AttributeError:
            mode = face_alignment.LandmarksType.TWO_D
        self.model = face_alignment.FaceAlignment(
            mode, device=device, flip_input=False, face_detector='sfd',
            face_detector_kwargs={"filter_threshold": 0.5}
        )

    @torch.no_grad()
    def __call__(self, image, with_landmarks=False):
        # image: 0-255, uint8, rgb, [3, h, w]
        out = self.model.get_landmarks(image.permute(1, 2, 0), detected_faces=None)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return None, None
            else:
                return None
        else:
            bbox, kpts = self.get_max_bbox(out*3)
            del out
            if with_landmarks:
                return bbox, kpts
            else:
                return bbox

    @staticmethod
    def get_max_bbox(kpts):
        def kpts_to_bbox(kpt):
            left, right = np.min(kpt[:, 0]), np.max(kpt[:, 0])
            top, bottom = np.min(kpt[:, 1]), np.max(kpt[:, 1])
            return np.array([left, top, right, bottom])
        all_bbox = torch.tensor(np.array([kpts_to_bbox(kpt.squeeze()) for kpt in kpts]))
        all_kpts = torch.tensor(np.array([kpt.squeeze() for kpt in kpts]))
        if all_kpts.shape[0] == 1:
            return all_bbox[0], all_kpts[0]
        boxes_size = (all_bbox[:, 3] - all_bbox[:, 1]) * (all_bbox[:, 2] - all_bbox[:, 0])
        max_idx = boxes_size.argmax()
        return all_bbox[max_idx], all_kpts[max_idx]
