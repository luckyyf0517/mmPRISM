import os
import glob
import json
import numpy as np
import argparse
import hashlib
import shutil

from tqdm import tqdm
from glob import glob
from termcolor import colored


def extract_collected_data(collected_root):
    """
    Extract files from collected folder structure to type-based folder structure
    
    Args:
        collected_root: Path to collected folder (e.g., '/root/autodl-tmp/datasets/tmp')
        output_root: Path to output folder (e.g., '/root/autodl-tmp/datasets/collected_extracted')
    """
    print(colored(f'Extracting data from {collected_root}', 'cyan'))
    
    # Define file types and their corresponding output folders (excluding pose.npy)
    file_types = {
        'color.npy': 'videos',
        # 'mmwave.npy': 'mmwave'
    }

    # Get all subdirectories in collected folder (0000, 0001, etc.)
    subdirs = sorted([d for d in os.listdir(collected_root) 
                     if os.path.isdir(os.path.join(collected_root, d))])
    
    print(colored(f'Found {len(subdirs)} subdirectories to process', 'yellow'))
    
    # Process each subdirectory
    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        subdir_path = os.path.join(collected_root, subdir)
        
        # Determine output root based on subfolder name
        if subdir.startswith('S'):
            current_output_root = '/root/autodl-tmp/datasets/collected_csl'
        else:
            print('Warning: DEBUG MODE')
            current_output_root = '/root/autodl-tmp/datasets/collected_demo'
            # current_output_root = '/root/autodl-tmp/datasets/collected_base'
        
        # Create output directory and subdirectories for current subfolder
        os.makedirs(current_output_root, exist_ok=True)
        for folder_name in file_types.values():
            os.makedirs(os.path.join(current_output_root, folder_name), exist_ok=True)
        
        # Get all files in the subdirectory
        files = os.listdir(subdir_path)
        
        for filename in files:
            file_path = os.path.join(subdir_path, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
                
            # Find matching file type
            target_folder = None
            for file_pattern, folder_name in file_types.items():
                if filename == file_pattern:
                    target_folder = folder_name
                    break
            
            # If file type is recognized, move to appropriate folder
            if target_folder:
                # Get file extension
                _, ext = os.path.splitext(filename)
                
                # Create new filename: subdir_number + extension
                new_filename = f"{subdir}{ext}"
                
                # Move file to target folder with new name
                src_path = file_path
                
                if target_folder == 'mmwave':
                    # For mmwave files, split into frames
                    mmwave_data = np.load(src_path)
                    seq_dir = os.path.join(current_output_root, target_folder, subdir)
                    os.makedirs(seq_dir, exist_ok=True)
                    
                    for frame_idx in range(mmwave_data.shape[0]):
                        frame_filename = f"{frame_idx:04d}.npy"
                        frame_path = os.path.join(seq_dir, frame_filename)
                        np.save(frame_path, mmwave_data[frame_idx])
                    print(colored(f'Split {filename} into {mmwave_data.shape[0]} frames in {subdir}', 'green'))
                else:
                    dst_path = os.path.join(current_output_root, target_folder, new_filename)
                    try:
                        if os.path.exists(dst_path):
                            os.remove(dst_path)
                        shutil.move(src_path, dst_path)
                        print(colored(f'Moved {filename} -> {target_folder}/{new_filename}', 'green'))
                    except Exception as e:
                        print(colored(f'Error moving {filename}: {e}', 'red'))
            else:
                print(colored(f'Unknown file type: {filename} in {subdir}', 'yellow'))
        
        # Remove the subdirectory after processing all files
        try:
            shutil.rmtree(subdir_path)
            print(colored(f'Removed subdirectory: {subdir}', 'blue'))
        except Exception as e:
            print(colored(f'Error removing subdirectory {subdir}: {e}', 'red'))
    
    print(colored(f'Extraction completed!', 'cyan'))
    
    # Print summary for both output directories
    for output_dir in ['/root/autodl-tmp/datasets/collected_csl', '/root/autodl-tmp/datasets/collected_base', '/root/autodl-tmp/datasets/collected_demo']:
        if os.path.exists(output_dir):
            print(colored(f'\nSummary for {output_dir}:', 'cyan'))
            for folder_name in file_types.values():
                folder_path = os.path.join(output_dir, folder_name)
                if os.path.exists(folder_path):
                    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
                    print(colored(f'{folder_name}: {file_count} files', 'blue'))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--collected_root', type=str, default='/root/autodl-tmp/datasets/tmp',
                       help='Root directory containing collected data with numbered subdirectories')
    args = parser.parse_args()

    # Extract collected data
    extract_collected_data(args.collected_root)
