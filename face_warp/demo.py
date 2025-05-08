import cv2
import dlib
import numpy as np
import os
import glob
import argparse
from imutils import face_utils
from scipy.spatial import procrustes

parser = argparse.ArgumentParser(description="Face Transfer with Structural Similarity Check")
parser.add_argument('--input', type=str, required=True, help='Input base directory containing subfolders')
args = parser.parse_args()

base_path = args.input.rstrip('/') 
original_frames_path = os.path.join(base_path, '1_initial')
new_head_frames_path = os.path.join(base_path, '2_headavatar')
output_frames_path = os.path.join(base_path, '3_face_warp')
shape_predictor_path = 'face_warp/shape_predictor_68_face_landmarks.dat'

# Structural Threshold
structure_threshold = 0.01
# Blending Factor
alpha =1.0

# Create output directory if it doesn't exist
if not os.path.exists(output_frames_path):
    os.makedirs(output_frames_path)

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Function to generate face mask from landmarks
def get_face_mask(image, landmarks):
    points = cv2.convexHull(landmarks)
    mask = np.zeros_like(image)
    cv2.fillConvexPoly(mask, points, (180, 180, 180))
    return mask

# Function to warp image based on affine transformation matrix
def warp_image(im, M, dshape):
    output_im = cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), flags=cv2.INTER_LINEAR)
    return output_im

# Function to read an image from file
def read_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")
    return image

# Function to detect facial landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None  # Return None if no face is detected
    shape = predictor(gray, rects[0])
    landmarks = face_utils.shape_to_np(shape)
    return landmarks

# Function to calculate structural similarity using Procrustes analysis
def calculate_structural_similarity(landmarks1, landmarks2):
    _, _, disparity = procrustes(landmarks1, landmarks2)
    return disparity

# Function for seamless cloning
def seamless_clone(src, dst, mask, center):
    output = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    return output

# Get sorted lists of original and new head images
original_images = sorted(glob.glob(f"{original_frames_path}/*.png"))
new_head_images = sorted(glob.glob(f"{new_head_frames_path}/*.png"))

# Process each frame
for i in range(len(original_images)):
    try:
        original_frame = read_image(original_images[i])

        if i <= 138:  # Adjust range if needed
            new_head_image = read_image(new_head_images[i])

            # Resize new head image if necessary
            if original_frame.shape != new_head_image.shape:
                new_head_image = cv2.resize(new_head_image, (original_frame.shape[1], original_frame.shape[0]))

            # Get landmarks for both images
            landmarks_orig = get_landmarks(original_frame)
            landmarks_new = get_landmarks(new_head_image)

            if landmarks_orig is None or landmarks_new is None:
                print(f"Frame {i+1}: No face detected, saving original frame.")
                cv2.imwrite(f"{output_frames_path}/{i:04d}.png", original_frame)
                continue

            # Check structural similarity between landmarks using Procrustes analysis
            structural_similarity = calculate_structural_similarity(landmarks_orig, landmarks_new)
            if structural_similarity > structure_threshold:
                print(f"Frame {i+1}: Structural similarity {structural_similarity:.2f} exceeds threshold, saving original frame.")
                cv2.imwrite(f"{output_frames_path}/{i:04d}.png", original_frame)
                continue

            # Compute affine transformation and warp the new head image
            M, _ = cv2.estimateAffinePartial2D(landmarks_new, landmarks_orig)
            warped_head = warp_image(new_head_image, M, original_frame.shape)

            # Blend the original and new head using weighted blending
            blended_image = cv2.addWeighted(warped_head, alpha, original_frame, 1 - alpha, 0)

            # Create face mask for seamless cloning
            mask = get_face_mask(blended_image, landmarks_orig)
            (x, y, w, h) = cv2.boundingRect(landmarks_orig)
            center = (x + w // 2, y + h // 2)
            output_frame = seamless_clone(blended_image, original_frame, mask, center)
            
            # Save the output frame
            cv2.imwrite(f"{output_frames_path}/{i:04d}.png", output_frame)
            print(f"Processed frame {i+1} with structural similarity {structural_similarity:.2f}")

        else:
            # If not in processing range, save the original frame as-is
            print(f"Frame {i+1}: No processing, saving original frame.")
            cv2.imwrite(f"{output_frames_path}/{i:04d}.png", original_frame)

    except Exception as e:
        print(f"Error processing frame {i+1}: {e}")
