import os
import sys
sys.path.append('demo/mmpose/projects/rtmpose3d')  

import cv2
import torch
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm
import time
import mmcv
import mmengine
import matplotlib.pyplot as plt
from mmengine.logging import print_log
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from scipy.ndimage import gaussian_filter1d


def process_single_image(frame, pose_estimator, args):
    """Process single image using RTMPose3D
    """
    h, w = frame.shape[:2]
    bbox = np.array([[0, 0, w, h]])  # Single bbox covering the whole frame
    
    # Estimate 3D pose
    pose_results = inference_topdown(pose_estimator, frame, bbox)
    return pose_results[0] 


def process_sequence(args, video_path, pose_estimator):
    """Process video sequence using RTMPose3D
    """

    save_path = video_path.replace('.mp4', '.npy').replace('videos', 'poses')
    if os.path.exists(save_path):
        print(colored(f'    [SKIP] {os.path.basename(video_path)}: {save_path} already exists', 'yellow'))
        return

    if not os.path.exists(video_path):
        print(colored(f'Video file {video_path} does not exist', 'red'))
        return
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    print(colored(f'\n>> Processing sequence: {seq_id}', 'cyan'))
    
    # Read frames
    print(colored('[1] Reading frames...', 'blue'))
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[20:, 20:-20]  # crop
        frames.append(frame)
    frames = np.array(frames)
    cap.release()
    print(colored(f'    [OK] Loaded {len(frames)} frames', 'green'))
    
    # Estimate poses
    print(colored('[2] Estimating poses...', 'blue'))
    keypoints_all = []
    for frame_idx in tqdm(range(len(frames)), desc='    Processing', ncols=80):
        frame = frames[frame_idx].copy()
        pose_results = process_single_image(frame, pose_estimator, args)
        keypoints = pose_results.pred_instances.keypoints[0]
        keypoints_all.append(keypoints)
        
    keypoints_all = np.array(keypoints_all)
    print(colored('    [OK] Pose estimation completed', 'green'))
    
    # Process keypoints
    print(colored('[3] Processing keypoints...', 'blue'))
    depths_center = keypoints_all[:, [6, 7], 2].mean()
    keypoints_all[..., 2] = keypoints_all[..., 2] - depths_center
    keypoints_all = np.concatenate([
        keypoints_all[:, :17], # body
        keypoints_all[:, -42:], # hands
    ], axis=1) # [N, 59, 3]

    def process_pose(pose):
        """Extract arm and hand joints from pose data"""
        return np.stack([
            np.concatenate([pose[:, [5,7,9], :], pose[:, -42:-21, :]], axis=-2),
            np.concatenate([pose[:, [6,8,10], :], pose[:, -21:, :]], axis=-2)
        ], axis=-3) # [N, 2, 24, 3]
    keypoints_all = process_pose(keypoints_all) # [N, 2, 24, 3]
    
    # Save results
    save_path = video_path.replace('.mp4', '.npy').replace('videos', 'poses')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    np.save(save_path, keypoints_all)
    print(colored(f'    [OK] Results saved to: {save_path}', 'green'))
    

def process_archive(args, archive_id, pose_estimator):
    """Process a single archive
    """
    base_path = '/root/autodl-tmp/datasets/csl-news'
    zip_path = os.path.join(base_path, f'archives/archive_{archive_id}.zip')
    extract_path = os.path.join(base_path, f'videos/archive_{archive_id}')

    if not os.path.exists(zip_path):
        print(colored(f'[X] Archive file {zip_path} does not exist', 'red'))
        return False
        
    # Extract archive
    print(colored('[1] Extracting archive...', 'blue'))
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(colored('    [OK] Archive extracted', 'green'))
    
    # Get all MP4 files
    mp4_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    mp4_files.sort()
    
    # Process each video
    total_videos = len(mp4_files)
    start_time = time.time()
    processed_videos = 0
    
    print(colored(f'[2] Processing {total_videos} videos...', 'blue'))
    for i, video_path in enumerate(mp4_files):
        if not ('Common-Concerns' in video_path or 'Dragon-TV' in video_path):
            print(colored(f'    [SKIP] {os.path.basename(video_path)}', 'yellow'))
            continue
        
        try: 
            process_sequence(args, video_path, pose_estimator)
            processed_videos += 1
        except Exception as e:
            print(colored(f'    [ERROR] {os.path.basename(video_path)}: {e}', 'red'))
            continue
        
        if processed_videos >= 2:
            avg_time_per_video = (time.time() - start_time) / processed_videos
            remaining_videos = total_videos - (i + 1)
            est_time_remaining = (remaining_videos * avg_time_per_video) / 60
            print(colored(f'[INFO] Progress: {i+1}/{total_videos}, ETA: {est_time_remaining:.1f}min', 'cyan'))
    
    # Cleanup
    print(colored('[3] Cleaning up...', 'blue'))
    import shutil
    shutil.rmtree(extract_path)
    print(colored('    [OK] Temporary files removed', 'green'))
    
    return True


def main():
    parser = argparse.ArgumentParser(description='RTMPose3D Body Pose Estimation')
    parser.add_argument('--id', nargs='+', default=None, help='List of sequences to process')
    parser.add_argument('--start', type=int, default=None, help='Start archive number')
    parser.add_argument('--end', type=int, default=None, help='End archive number')
    parser.add_argument('--gpu', type=int, dest='gpu_id', help='GPU ID')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint threshold')
    args = parser.parse_args()
    
    if args.id is None:
        args.id = list(range(args.start, args.end))
        args.id = [f'{i:03d}' for i in args.id]
    args.device = 'cuda:' + str(args.gpu_id)    

    # Initialize model
    print(colored('Initializing model...', 'blue'))
    pose_config = 'demo/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py'
    pose_checkpoint = 'demo/mmpose/projects/rtmpose3d/demo/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'
    
    pose_estimator = init_model(pose_config, pose_checkpoint, device=args.device)
    print(colored('Model loaded successfully', 'green'))
    
    # Process each archive
    failed_seq_list = []
    for archive_id in args.id:
        print(colored(f'\nProcessing archive_{archive_id}', 'blue'))
        print(colored('=' * 50, 'blue'))
        
        success = process_archive(args, archive_id, pose_estimator)
        
        if success:
            print(colored(f'Successfully completed archive_{archive_id}', 'green'))
        else:
            failed_seq_list.append(archive_id)
            print(colored(f'Failed to process archive_{archive_id}', 'red'))
        
        print(colored('=' * 50, 'blue'))

    if len(failed_seq_list) > 0:
        print(colored(f'Failed archives: {failed_seq_list}', 'red'))
    else:
        print(colored('\nAll archives processed successfully', 'green'))

if __name__ == '__main__':
    main()