import zipfile
import os
import shutil
from tqdm import tqdm

def extract_zip_preserve(zip_filepath, target_dir='.'):

    temp_dir = os.path.join(target_dir, 'temp_extract')
    try:
        os.makedirs(temp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            
            print(f"Extracting Checkpoints & Data ...")
            for file in tqdm(zip_ref.namelist(), total=total_files, desc="Extracting", unit="files"):
                zip_ref.extract(file, temp_dir)
        avatar_dir = None
        for item in os.listdir(temp_dir):
            if item == 'Avatar' and os.path.isdir(os.path.join(temp_dir, item)):
                avatar_dir = os.path.join(temp_dir, item)
                break
            
        subdirs = ['avatar', 'face_warp', 'fitting', 'submodules']
        total_files_to_process = 0
        files_to_process = []
        
        for subdir in subdirs:
            source_dir = os.path.join(avatar_dir, subdir)
            if os.path.exists(source_dir):
                for root, _, files in os.walk(source_dir):
                    rel_path = os.path.relpath(root, source_dir)
                    dest_path = os.path.join(target_dir, subdir, rel_path) if rel_path != '.' else os.path.join(target_dir, subdir)
                    
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(dest_path, file)
                        files_to_process.append((src_file, dst_file, dest_path))
                        total_files_to_process += 1
    
        print(f"Adding files into {target_dir}...")
        skipped = 0
        added = 0
        
        for src_file, dst_file, dest_path in tqdm(files_to_process, total=total_files_to_process, desc="Merging", unit="files"):
            os.makedirs(dest_path, exist_ok=True)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                added += 1
            else:
                skipped += 1
        
        print(f"Extraction complete!")
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"Error during extraction and merging: {e}")
        if os.path.exists(temp_dir):
            print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

zip_path = './Avatar.zip'
base_dir = './'
extract_zip_preserve(zip_path, base_dir)