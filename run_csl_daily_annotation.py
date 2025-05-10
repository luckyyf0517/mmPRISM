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


def process_sequence(args, seq_folder, pose_estimator):
    """Process image sequence using RTMPose3D
    """
    seq_id = os.path.basename(seq_folder)
    save_path = f'/root/autodl-tmp/datasets/CSL-Daily/sentence/poses/{seq_id}.npy'
    
    if os.path.exists(save_path) and not args.force:
        print(colored(f'    [SKIP] {seq_id}: {save_path} already exists', 'yellow'))
        return

    print(colored(f'\n>> Processing sequence: {seq_id}', 'cyan'))
    
    # Read frames
    print(colored('[1] Reading frames...', 'blue'))
    image_files = sorted([f for f in os.listdir(seq_folder) if f.endswith('.jpg')])
    frames = []
    for img_file in image_files:
        frame = cv2.imread(os.path.join(seq_folder, img_file))
        if frame is not None:
            frames.append(frame)
    
    frames = np.array(frames)
    print(colored(f'    [OK] Loaded {len(frames)} frames', 'green'))
    
    # Estimate poses
    print(colored('[2] Estimating poses...', 'blue'))
    pose_results_all = []
    for frame_idx in tqdm(range(len(frames)), desc='    Processing', ncols=80):
        frame = frames[frame_idx].copy()
        pose_results = process_single_image(frame, pose_estimator, args)
        pose_results_all.append(pose_results)
        
    print(colored('    [OK] Pose estimation completed', 'green'))
    
    # Process keypoints
    print(colored('[3] Processing keypoints...', 'blue'))
    # Extract keypoints for processing
    keypoints_all = []
    keypointscores_all = []
    for pose_result in pose_results_all:
        keypoints = pose_result.pred_instances.keypoints[0]
        keypointscores = pose_result.pred_instances.keypoint_scores[0]
        
        # Set low confidence keypoints to NaN
        low_conf_mask = keypointscores < args.kpt_thr
        keypoints[low_conf_mask] = np.nan
        
        keypoints_all.append(keypoints)
        keypointscores_all.append(keypointscores)
    
    keypoints_all = np.array(keypoints_all)
    keypointscores_all = np.array(keypointscores_all)
    
    # Process depth center
    depths_center = keypoints_all[:, [6, 7], 2].mean()
    keypoints_all[..., 2] = keypoints_all[..., 2] - depths_center
    
    # Extract body and hand keypoints
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
    processed_keypoints = process_pose(keypoints_all) # [N, 2, 24, 3]

    # Check if arm keypoints (first 3 points) contain NaN values
    left_arm_has_nan = np.isnan(processed_keypoints[:, 0, :3]).any()
    right_arm_has_nan = np.isnan(processed_keypoints[:, 1, :3]).any()

    # Check if hand keypoints (last 21 points) at least one point is not NaN
    left_hand_all_nan = np.isnan(processed_keypoints[:, 0, -21:]).all()
    right_hand_all_nan = np.isnan(processed_keypoints[:, 1, -21:]).all()
    
    # Skip sequence if either arm contains NaN values
    if left_arm_has_nan or right_arm_has_nan or left_hand_all_nan or right_hand_all_nan:
        print(colored(f'    [SKIP] {seq_id}: Arm keypoints contain NaN values'
                      f'(left arm has NaN: {left_arm_has_nan}, '
                      f'right arm has NaN: {right_arm_has_nan}, '
                      f'left hand all NaN: {left_hand_all_nan}, '
                      f'right hand all NaN: {right_hand_all_nan})', 'yellow'))
        return
    
    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    np.save(save_path, processed_keypoints)
    print(colored(f'    [OK] Results saved to: {save_path}', 'green'))

def process_dataset(args, pose_estimator):
    """Process the CSL-Daily dataset
    """
    base_path = '/root/autodl-tmp/datasets/CSL-Daily/sentence/images'
    
    # Get all sequence folders
    seq_folders = []
    for item in os.listdir(base_path):
        folder_path = os.path.join(base_path, item)
        if os.path.isdir(folder_path):
            seq_folders.append(folder_path)
    
    seq_folders.sort()
    
    # Process each sequence folder
    total_sequences = len(seq_folders)
    start_time = time.time()
    processed_sequences = 0
    
    print(colored(f'Processing all {total_sequences} sequences...', 'blue'))
    for i, seq_folder in enumerate(seq_folders):
        try: 
            process_sequence(args, seq_folder, pose_estimator)
            processed_sequences += 1
        except Exception as e:
            print(colored(f'    [ERROR] {os.path.basename(seq_folder)}: {e}', 'red'))
            continue
        
        if processed_sequences >= 2:
            avg_time_per_seq = (time.time() - start_time) / processed_sequences
            remaining_seqs = total_sequences - (i + 1)
            est_time_remaining = (remaining_seqs * avg_time_per_seq) / 60
            print(colored(f'[INFO] Progress: {i+1}/{total_sequences}, ETA: {est_time_remaining:.1f}min', 'cyan'))
    
    return True


def main():
    parser = argparse.ArgumentParser(description='RTMPose3D Body Pose Estimation for CSL-Daily')
    parser.add_argument('--gpu', type=int, dest='gpu_id', help='GPU ID')
    parser.add_argument('--kpt-thr', type=float, default=0.5, help='Keypoint threshold')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of existing results')
    args = parser.parse_args()
    
    args.device = 'cuda:' + str(args.gpu_id)    

    # Initialize model
    print(colored('Initializing model...', 'blue'))
    pose_config = 'demo/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py'
    pose_checkpoint = 'demo/mmpose/projects/rtmpose3d/demo/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'
    
    pose_estimator = init_model(pose_config, pose_checkpoint, device=args.device)
    print(colored('Model loaded successfully', 'green'))
    
    # Process the dataset
    success = process_dataset(args, pose_estimator)
    
    if success:
        print(colored('Dataset processing completed successfully', 'green'))
    else:
        print(colored('Failed to process dataset', 'red'))

if __name__ == '__main__':
    main()