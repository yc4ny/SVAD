### 1. Video Diffusion Model for Generating Initial Frames
Below, we give example commands  with a single image from the Snapshot Human dataset sequence "female-3-casual". Match the format of folder, file structures and use your images!  

Go to MusePose submodule directory: 
```bash
cd submodules/MusePose
```
Copy initial input image to MusePose directory:
```bash
cp ../../data/female-3-casual/ref.png assets/images/
```

Get the aligned dwpose of the reference image. Use the dance video we have provided: 
```bash
python pose_align.py --imgfn_refer assets/images/ref.png --vidfn assets/videos/dance.mp4
```
Create a .yaml file for generation config in *configs* folder with the following content: 

**config/female-3-casual.yaml**
```bash
pretrained_base_model_path: './pretrained_weights/sd-image-variations-diffusers'
pretrained_vae_path: './pretrained_weights/sd-vae-ft-mse'
image_encoder_path: './pretrained_weights/image_encoder'

denoising_unet_path: "./pretrained_weights/MusePose/denoising_unet.pth"
reference_unet_path: "./pretrained_weights/MusePose/reference_unet.pth"
pose_guider_path: "./pretrained_weights/MusePose/pose_guider.pth"
motion_module_path: "./pretrained_weights/MusePose/motion_module.pth"

inference_config: "./configs/inference_v2.yaml"
weight_dtype: 'fp16'

test_cases:
  "./assets/images/ref.png": # Path to initial input single image
    - "./assets/poses/align/img_ref_video_dance.mp4" # Path to the aligned dwpose of the reference image in the step just before. 

```
Run video diffusion model inference:
```bash
python test_stage_2.py --config configs/female-3-casual.yaml
```
Move to base directory and extract the ouput video to image frames: 
```bash
cd ../../
python utils/extract_frames.py --input submodules/MusePose/output/initial.mp4 --output data/female-3-casual/1_initial/
```

### 2. Create 3D Head Renderings

Move to GAGAvatar directory:
```bash
cd submodules/GAGAvatar
```

Run inference on GAGAvatar to generate 3d head avatar. We have already provided a driving sequence (driver): 
```bash
python inference_single.py -d drivers/sage/ -i ../../data/female-3-casual/ref.png -o ../../data/female-3-casual/2_headavatar
cd ../../ # Return to base directory after inference
```

### 3. Face Warping 
Perform facewarping to enhance facial details of the original diffusion output with the 3D head renderings:
```bash
python face_warp/demo.py --input data/female-3-casual/
```

### 4. Face Restoration 
Move to BFRffusion directory:
```bash
cd submodules/BFRffusion
```
Apply super resolution:
```bash
python inference.py --input ../../data/female-3-casual/3_face_warp --output ../../data/female-3-casual/4_superres --sr_scale 2 
# You may adjust parameter "sr_scale" (super resolution scale). 
# Values too high lead to shiny features in the face and take longer to train. 
cd ../../
```

## Train 3DGS Avatar with Generated Synthetic Data


### Run SMPL-X Fitting
Copy the synthetic training data to *fitting/data/Custom/data/* folder:
```bash
mkdir -p fitting/data/Custom/data/female-3-casual/frames/
cp -r data/female-3-casual/4_superres/* fitting/data/Custom/data/female-3-casual/frames/
```

Move to *fitting/tools/* directory and create the train/test/splits:
```bash
cd fitting/tools/ 
python create_frame_list.py --input ../data/Custom/data/female-3-casual/
# This will create the frame_list_all.txt, frame_list_train.txt, frame_list_test.txt in the input folder directory.
# Necessary for fitting/training avatar.  
```

Start Avatar Fitting:
```bash
CUDA_VISIBLE_DEVICES=0  python run.py --root_path ../data/Custom/data/female-3-casual/
# You must specify GPU ID or there will be an error. 
# Set CUDA_VISIBLE_DEVICES={gpu_id} when running fitting. 
```
### Run Avatar Training 

Move preprocessed files to training folder:
```bash
mv ../data/Custom/data/female-3-casual/ ../../avatar/data/Custom/data/
```
Move to the directory where avatar is trained: 
```bash
cd ../../avatar/main
```

Start Training Avatar: 
```bash
python train.py --subject_id female-3-casual
```