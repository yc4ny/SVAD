import os
import argparse
import random

def get_image_list(input_folder):
    images = [img for img in sorted(os.listdir(os.path.join(input_folder,"frames"))) if img.endswith('.png')]
    return images

def write_file(filepath, content):
    with open(filepath, 'w') as f:
        for item in content:
            f.write(f"{item}\n")

def create_frame_lists(input_folder):
    # Save directly to the input folder instead of the parent
    images = get_image_list(input_folder)
    frame_list_all = os.path.join(input_folder, 'frame_list_all.txt')
    write_file(frame_list_all, images)
    print(f"Saved all frames to {frame_list_all}")

    random.shuffle(images)
    split_index = int(0.9 * len(images))
    train_images = images[:split_index]
    test_images = images[split_index:]

    frame_list_train = os.path.join(input_folder, 'frame_list_train.txt')
    frame_list_test = os.path.join(input_folder, 'frame_list_test.txt')
    write_file(frame_list_train, train_images)
    write_file(frame_list_test, test_images)
    
    print(f"Saved train frames to {frame_list_train}")
    print(f"Saved test frames to {frame_list_test}")

def main():
    parser = argparse.ArgumentParser(description="Generate frame lists from image folder")
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing images')
    args = parser.parse_args()
    create_frame_lists(args.input)

if __name__ == '__main__':
    main()