import os
import glob
import json
import torch
import argparse
import torchvision
from tqdm.rich import tqdm

from engines.utils_lmdb import LMDBEngine
from engines.engine_core import expand_bbox
from engines.vgghead_detector import VGGHeadDetector
from engines.human_matting import StyleMatteEngine as HumanMattingEngine

class FaceDetector:
    def __init__(self, device='cuda'):
        self._device = device
        self.vgg_detector = VGGHeadDetector(device=device)
        self.matting_engine = HumanMattingEngine(device=device)

    def forward(self, image_path):
        inp_image = torchvision.io.read_image(image_path, mode=torchvision.io.image.ImageReadMode.RGB)
        inp_image = inp_image.to(self._device).float()
        _, bbox, _ = self.vgg_detector(inp_image, 'online_track')
        if bbox is None:
            print(f'No face detected in {image_path}.')
            return None
        bbox = expand_bbox(bbox, scale=1.65).long()
        crop_image = torchvision.transforms.functional.crop(
            inp_image, top=bbox[1], left=bbox[0], height=bbox[3]-bbox[1], width=bbox[2]-bbox[0]
        )
        crop_image = torchvision.transforms.functional.resize(crop_image, (512, 512), antialias=True)
        crop_image = self.matting_engine(crop_image/255.0, return_type='matting', background_rgb=0.0).cpu().clamp(0.0, 1.0) * 255.0
        return crop_image


if __name__ == '__main__':
    # video_dirs is the list of video directories, each contains frames of the video, named as [frameid.jpg].
    print('This is a helper script to build LMDB database for training.')
    print('Please make sure you have the correct path to your storage and have modified the code.')
    print('PATH_TO_YOUR_STORAGE should be your dump path.')
    print('VIDEO_DIRS should be a list of your video directories, the names should not be too long, preferably INT numbers')
    data_processor = FaceDetector()
    lmdb_engine = LMDBEngine(PATH_TO_YOUR_STORAGE, write=True)
    for vidx, video_dir_path in enumerate(VIDEO_DIRS):
        frames = glob.glob(f'{video_dir_path}/*.jpg') # image name shuold be [frameid.jpg]
        frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        video_id = os.path.basename(video_dir_path[:-1] if video_dir_path.endswith('/') else video_dir_path) # video_id should be INT
        for frame_path in tqdm(frames, desc=f'Processing video {vidx+1}/{len(VIDEO_DIRS)}:'):
            frame_tensor = detector.forward(frame_path)
            frame_id = int(os.path.splitext(os.path.basename(frame_path))[0])
            if frame_tensor is None:
                continue
            frame_tensor = frame_tensor.to(torch.uint8).cpu()
            # torchvision.io.write_jpeg(frame_tensor.to(torch.uint8).cpu(), 'debug.jpg', quality=90)
            dump_name = '{:06d}_{}'.format(video_id, frame_id)
            if lmdb_engine.exists(dump_name):
                print('Frame already exists: {}'.format(frame_path))
                continue
            if image_tensor.max() < 5.0:
                print('Frame empty: {}'.format(frame_path))
                continue
            # print(dump_name)
            lmdb_engine.dump(dump_name, payload=frame_tensor, type='image')
    lmdb_engine.random_visualize(os.path.join(PATH_TO_YOUR_STORAGE, 'visualize.jpg'))
    lmdb_engine.close()
