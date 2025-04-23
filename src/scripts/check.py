import os
import glob
import numpy as np
from tqdm import tqdm
from termcolor import colored

def check_and_clean_poses(poses_root):
    """
    Check all pose files and remove those that don't match expected shape [N, 57, 3].
    Provides detailed logging of the process.
    
    Args:
        poses_root (str): Root directory containing pose files
    """
    # Collect all pose files
    pose_files = glob.glob(os.path.join(poses_root, 'archive_*/*.npy'))
    print(colored(f'Found {len(pose_files)} pose files to check', 'cyan'))
    
    removed_files = []
    invalid_shapes = []
    
    for pose_path in tqdm(pose_files, desc='Checking poses'):
        try:
            pose = np.load(pose_path)
            if len(pose.shape) != 3 or pose.shape[1:] != (59, 3):
                print(colored(f'Invalid shape {pose.shape} in file: {pose_path}', 'yellow'))
                invalid_shapes.append(f'{pose_path}: {pose.shape}')
                # os.remove(pose_path)
                removed_files.append(pose_path)
        except Exception as e:
            print(colored(f'Error loading file {pose_path}: {str(e)}', 'red'))
            # os.remove(pose_path)
            removed_files.append(pose_path)
            
    # Print summary
    print('\nCheck completed!')
    print(colored(f'Total files checked: {len(pose_files)}', 'green'))
    print(colored(f'Files removed: {len(removed_files)}', 'red'))
    
    if invalid_shapes:
        print('\nInvalid shapes found:')
        for item in invalid_shapes:
            print(colored(item, 'yellow'))
            
    if removed_files:
        print('\nRemoved files:')
        for file in removed_files:
            print(colored(file, 'red'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses_root', type=str, default='/root/autodl-tmp/datasets/csl-news/poses',
                       help='Root directory containing pose files')
    args = parser.parse_args()
    
    check_and_clean_poses(args.poses_root)
