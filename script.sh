#!/usr/bin/env bash

# Avatar Generation Automation Script 
# Usage: bash script.sh <name_of_image> <gpu_id>
# Example: bash script.sh female-3-casual 0

# Make sure this script runs with bash 
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash to run. Please use:"
    echo "bash $0 $*"
    exit 1
fi

# Record overall start time
TOTAL_START_TIME=$(date +%s)

# Check if the required arguments were provided
if [ $# -lt 2 ]; then
    echo "Error: Please provide the name of the avatar and GPU ID."
    echo "Usage: bash $0 <name_of_avatar> <gpu_id>"
    exit 1
fi

# Set the image name and GPU ID from the arguments
IMAGE_NAME=$1
GPU_ID=$2

# Check if the input image exists
if [ ! -f "data/${IMAGE_NAME}/ref.png" ]; then
    echo "Error: Input image not found at data/${IMAGE_NAME}/ref.png"
    echo "Please place your input image at data/${IMAGE_NAME}/ref.png"
    exit 1
fi

echo "Starting avatar generation for: ${IMAGE_NAME} using GPU: ${GPU_ID}"
echo "-------------------------------------------"

# Create necessary directories
mkdir -p data/${IMAGE_NAME}/1_initial/
mkdir -p data/${IMAGE_NAME}/2_headavatar/
mkdir -p data/${IMAGE_NAME}/3_face_warp/
mkdir -p data/${IMAGE_NAME}/4_superres/

# Initialize timing arrays
declare -a STEP_NAMES
declare -a STEP_TIMES

# Step 1. Video Diffusion Model 
echo "************************Step 1: Video Diffusion Model for Generating Initial Frames************************"
STEP1_START_TIME=$(date +%s)

# Create subject-specific directories for pose alignment
mkdir -p submodules/MusePose/assets/images/${IMAGE_NAME}
mkdir -p submodules/MusePose/assets/poses/align/${IMAGE_NAME}
mkdir -p submodules/MusePose/output/${IMAGE_NAME}

# Go to MusePose directory and copy input image
cd submodules/MusePose
cp ../../data/${IMAGE_NAME}/ref.png assets/images/${IMAGE_NAME}/

# Get the aligned dwpose of the reference image (use subject-specific paths)
echo "Getting aligned dwpose of the reference image..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python pose_align.py --imgfn_refer assets/images/${IMAGE_NAME}/ref.png --vidfn assets/videos/dance.mp4 --outfn assets/poses/align/${IMAGE_NAME}/img_ref_video_dance.mp4 --outfn_align_pose_video assets/poses/align/${IMAGE_NAME}/img_ref_video_dance_pose.mp4

# Create config file for generation with subject-specific paths
echo "Creating configuration file..."
cat > configs/${IMAGE_NAME}.yaml << EOF
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
  "./assets/images/${IMAGE_NAME}/ref.png": # Path to initial input single image
    - "./assets/poses/align/${IMAGE_NAME}/img_ref_video_dance_pose.mp4" # Path to the aligned dwpose of the reference image
EOF

# Run video diffusion model inference with subject-specific output
echo "Running video diffusion model inference..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python test_stage_2.py --config configs/${IMAGE_NAME}.yaml --image_name ${IMAGE_NAME}

# Move back to base directory and extract output video frames
cd ../../
echo "Extracting frames from output video..."
# Use the reference image name for the video file
python utils/extract_frames.py --input submodules/MusePose/output/${IMAGE_NAME}.mp4 --output data/${IMAGE_NAME}/1_initial/

STEP1_END_TIME=$(date +%s)
STEP1_DURATION=$((STEP1_END_TIME - STEP1_START_TIME))
STEP_NAMES+=("Video Diffusion Model")
STEP_TIMES+=($STEP1_DURATION)

# Step 2. Identity Preservation Module (now includes face warping)
echo "************************Step 2: Identity Preservation Module************************"
STEP2_START_TIME=$(date +%s)

cd submodules/GAGAvatar

# Clean any existing output to prevent identity mixing
mkdir -p ../../data/${IMAGE_NAME}/2_headavatar
rm -rf ../../data/${IMAGE_NAME}/2_headavatar/*

# Create subject-specific tracking directories
mkdir -p render_results/tracked/${IMAGE_NAME}

# Create subject-specific driver directory
mkdir -p drivers/sage_${IMAGE_NAME}
if [ -d "drivers/sage" ]; then
    echo "Copying driver files to subject-specific directory..."
    cp -r drivers/sage/* drivers/sage_${IMAGE_NAME}/ 2>/dev/null || true
fi

# Run inference on GAGAvatar with subject-specific output and force retrack to ensure correct identity
echo "Running GAGAvatar inference..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_single.py -d drivers/sage_${IMAGE_NAME}/ -i ../../data/${IMAGE_NAME}/ref.png -o ../../data/${IMAGE_NAME}/2_headavatar --gpu ${GPU_ID} --force_retrack
cd ../../

# Face Warping
echo "Performing face warping..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python face_warp/demo.py --input data/${IMAGE_NAME}/

STEP2_END_TIME=$(date +%s)
STEP2_DURATION=$((STEP2_END_TIME - STEP2_START_TIME))
STEP_NAMES+=("Identity Preservation Module")
STEP_TIMES+=($STEP2_DURATION)

# Step 3. Image Restoration Module
echo "************************Step 3: Image Restoration Module************************"
STEP3_START_TIME=$(date +%s)

# Super Resolution
echo "Applying super resolution..."
cd submodules/BFRffusion
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference.py --input ../../data/${IMAGE_NAME}/3_face_warp --output ../../data/${IMAGE_NAME}/4_superres --sr_scale 2
cd ../../

# # Copy synthetic training data
# echo "Copying synthetic training data..."
mkdir -p fitting/data/Custom/data/${IMAGE_NAME}/frames/
cp -r data/${IMAGE_NAME}/4_superres/* fitting/data/Custom/data/${IMAGE_NAME}/frames/

STEP3_END_TIME=$(date +%s)
STEP3_DURATION=$((STEP3_END_TIME - STEP3_START_TIME))
STEP_NAMES+=("Image Restoration Module")
STEP_TIMES+=($STEP3_DURATION)

# Step 4. Run SMPL-X Fitting
echo "************************Step 4: SMPL-X Fitting************************"
STEP4_START_TIME=$(date +%s)

cd fitting/tools/
# Create train/test/val .txt files
python create_frame_list.py --input ../data/Custom/data/${IMAGE_NAME}

# Start Avatar Fitting with updated arguments for parallel processing
echo "Starting avatar fitting..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python run.py --root_path ../data/Custom/data/${IMAGE_NAME}/ --gpu ${GPU_ID}

# Move fitting data
echo "Moving preprocessed files to training folder..."
mkdir -p ../../avatar/data/Custom/data/
cp -r ../data/Custom/data/${IMAGE_NAME}/ ../../avatar/data/Custom/data/

STEP4_END_TIME=$(date +%s)
STEP4_DURATION=$((STEP4_END_TIME - STEP4_START_TIME))
STEP_NAMES+=("SMPL-X Fitting")
STEP_TIMES+=($STEP4_DURATION)

# Step 5. Run Avatar Training
echo "************************Step 5: Avatar Training************************"
STEP5_START_TIME=$(date +%s)

cd ../../avatar/main

# Start training avatar 
echo "Starting avatar training..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py --subject_id ${IMAGE_NAME}

STEP5_END_TIME=$(date +%s)
STEP5_DURATION=$((STEP5_END_TIME - STEP5_START_TIME))
STEP_NAMES+=("Avatar Training")
STEP_TIMES+=($STEP5_DURATION)

# Calculate total time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

# Display timing summary
echo ""
echo "====================================================="
echo "          AVATAR GENERATION TIMING SUMMARY           "
echo "====================================================="
echo "Avatar ID: ${IMAGE_NAME}, GPU: ${GPU_ID}"
echo "-----------------------------------------------------"
echo "Step                    | Duration (Min:Sec)"
echo "-----------------------------------------------------"

# Display timing for each step
for i in "${!STEP_NAMES[@]}"; do
    MINS=$((STEP_TIMES[$i] / 60))
    SECS=$((STEP_TIMES[$i] % 60))
    printf "%-24s | %02d:%02d\n" "${STEP_NAMES[$i]}" $MINS $SECS
done

echo "-----------------------------------------------------"
# Total time
TOTAL_MINS=$((TOTAL_DURATION / 60))
TOTAL_SECS=$((TOTAL_DURATION % 60))
printf "%-24s | %02d:%02d\n" "TOTAL TIME" $TOTAL_MINS $TOTAL_SECS
echo "====================================================="
echo ""

echo "Avatar generation complete for: ${IMAGE_NAME}"
echo "Results saved to: ${OUTPUT_DIR}"
