import os
import zipfile
from tqdm import tqdm
from termcolor import colored
from glob import glob

def zip_folder(folder_path, output_zip):
    """Compress folder to zip file"""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                zipf.write(file_path, arcname)

def main():
    # Get all folders starting with 'archive_' in current directory
    folders = glob('/root/autodl-tmp/datasets/csl-news/poses/archive_*/')
    folders.sort()  # Sort to ensure order
    
    print(colored(f"\nFound {len(folders)} folders to compress", "cyan"))
    
    # Compress each folder
    for folder in tqdm(folders, desc=colored("Compressing folders", "blue")):
        folder = folder.rstrip('/')  # Remove trailing slash
        zip_path = f"/root/autodl-tmp/datasets/csl-news/poses_zip/{os.path.basename(folder)}.zip"
        zip_folder(folder, zip_path)

if __name__ == "__main__":
    main() 