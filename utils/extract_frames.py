import argparse
import cv2
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video and save as images.')
    parser.add_argument('--input', required=True, help='Path to input video file.')
    parser.add_argument('--output', required=True, help='Path to output directory.')

    args = parser.parse_args()

    input_video_path = args.input
    output_dir = args.output

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get total frame count for the progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create progress bar
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        frame_count = 0
        success, frame = cap.read()

        while success:
            # Format frame number with leading zeros
            frame_number_str = f'{frame_count:04d}'
            output_path = os.path.join(output_dir, f'{frame_number_str}.png')

            # Save frame as PNG image with no compression (best quality)
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            frame_count += 1
            success, frame = cap.read()
            
            # Update progress bar
            pbar.update(1)

    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")

if __name__ == '__main__':
    main()