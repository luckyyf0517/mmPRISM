import os
import zipfile
from tqdm import tqdm
from termcolor import colored
from glob import glob
import shutil

def unzip_folder(zip_path, output_folder):
    """Unzip zip file to output_folder, flattening the directory structure"""
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # Get list of file names in zip
        file_list = zipf.namelist()
        
        # Extract each file directly to output folder
        for file in file_list:
            # Skip directories
            if file.endswith('/'):
                continue
                
            # Get the base filename without any path
            filename = os.path.basename(file)
            
            # Extract the file
            with zipf.open(file) as source, open(os.path.join(output_folder, filename), 'wb') as target:
                shutil.copyfileobj(source, target)

def main():
    zip_dir = '/root/autodl-tmp/datasets/csl-news/poses_zip/'
    out_dir = '/root/autodl-tmp/datasets/csl-news/poses/'
    zip_files = glob(os.path.join(zip_dir, '*.zip'))
    zip_files.sort()

    print(colored(f"\nFound {len(zip_files)} zip files to check and extract", "cyan"))

    for zip_file in tqdm(zip_files, desc=colored("Extracting zips", "blue")):
        folder_name = os.path.splitext(os.path.basename(zip_file))[0]
        target_folder = os.path.join(out_dir, folder_name)
        if not os.path.exists(target_folder):
            print(colored(f"Extracting {zip_file} to {target_folder}", "green"))
            os.makedirs(target_folder, exist_ok=True)
            unzip_folder(zip_file, target_folder)
        else:
            print(colored(f"Skip {folder_name}, already exists.", "yellow"))

if __name__ == "__main__":
    main() 