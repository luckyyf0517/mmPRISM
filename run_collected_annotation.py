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
from src.utils.plot import plot_hand_cv2


def process_single_image(frame, pose_estimator, args):
    """Process single image using RTMPose3D
    """
    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    bbox = np.array([[0, 0, w, h]])  # Single bbox covering the whole frame
    
    # Estimate 3D pose
    pose_results = inference_topdown(pose_estimator, frame, bbox)
    return pose_results[0] 


def process_sequence(args, video_path, pose_estimator):
    """Process video sequence using RTMPose3D
    """
    # Get sequence ID from path
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Define save path
    save_path = video_path.replace('videos', 'poses')
    # if os.path.exists(save_path):
    #     print(colored(f'    [SKIP] {seq_id}: {save_path} already exists', 'yellow'))
    #     return

    if not os.path.exists(video_path):
        print(colored(f'Video file {video_path} does not exist', 'red'))
        return
    print(colored(f'\n>> Processing sequence: {seq_id}', 'cyan'))
    
    # Read frames from npy file
    print(colored('[1] Reading frames...', 'blue'))
    frames = np.load(video_path)  # [N, 640, 360, 3]
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
    center = keypoints_all[:, [6, 7], :].mean(axis=1, keepdims=True)
    keypoints_all = keypoints_all - center

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
    
    # Create poses directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save results
    np.save(save_path, keypoints_all.astype(np.float32))
    print(colored(f'    [OK] Results saved to: {save_path}', 'green'))
    
    # Save annotation video if requested
    if args.save_video:
        print(colored('[4] Creating annotation video...', 'blue'))
        video_save_path = video_path.replace('videos', 'annotation').replace('npy', 'mp4')
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        
        camera_params = {
            'camera_position': [0, 0, -1000],
            'camera_intrinsic': [800, 800, 256.0, 256.0]
        }
        
        # Setup video writer with H.264 codec for VSCode compatibility
        # Try different codecs in order of preference
        codecs = ['mp4v', 'avc1', 'XVID']
        out = None
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(video_save_path, fourcc, 30.0, (512, 512))
                if out.isOpened():
                    break
            except:
                continue
        # Interpolate keypoints from 10fps to 30fps using scipy
        from scipy.interpolate import interp1d
        
        # Create time points for original and interpolated sequences
        t_orig = np.arange(len(keypoints_all))
        t_interp = np.linspace(0, len(keypoints_all)-1, len(keypoints_all)*3)
        # Reshape keypoints for interpolation [N,2,24,3] -> [N, 144]
        keypoints_flat = keypoints_all.reshape(len(keypoints_all), -1)
        # Create interpolation function
        interp_func = interp1d(t_orig, keypoints_flat, axis=0, kind='linear')
        # Get interpolated keypoints and reshape back
        keypoints_interp = interp_func(t_interp).reshape(-1, 2, 24, 3)
                
        # Write frames with annotations
        for frame_idx in tqdm(range(len(keypoints_interp)), desc='    Writing video', ncols=80):
            joints_l = np.concatenate([
                keypoints_interp[frame_idx, 0, :3],  # left arm 3 points
                keypoints_interp[frame_idx, 0, 3:]   # left hand 21 points
            ])
            joints_r = np.concatenate([
                keypoints_interp[frame_idx, 1, :3],  # right arm 3 points
                keypoints_interp[frame_idx, 1, 3:]   # right hand 21 points
            ])
            
            # plot hand
            frame_vis = plot_hand_cv2(
                camera_params=camera_params,
                joints_l=joints_l * 2.0e2,
                joints_r=joints_r * 2.0e2
            )
            
            # Write frame
            out.write(frame_vis)
            
        # Release video writer
        out.release()
        print(colored(f'    [OK] Video saved to: {video_save_path}', 'green'))


def process_archive(args, archive_id, pose_estimator):
    """Process a single archive
    """
    video_path = os.path.join(args.base_path, 'videos', f'{archive_id}.npy')
    
    try:
        process_sequence(args, video_path, pose_estimator)
        return True
    except Exception as e:
        print(colored(f'    [ERROR] {archive_id}: {e}', 'red'))
        return False


def main():
    parser = argparse.ArgumentParser(description='RTMPose3D Body Pose Estimation')
    parser.add_argument('--id', nargs='+', default=None, help='List of sequences to process')
    parser.add_argument('--start', type=int, default=None, help='Start archive number')
    parser.add_argument('--end', type=int, default=None, help='End archive number')
    parser.add_argument('--gpu', type=int, dest='gpu_id', help='GPU ID')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint threshold')
    parser.add_argument('--save-video', action='store_true', help='Save video with annotations')
    parser.add_argument('--base-path', type=str, default='data/collected_base', help='Base path for data')
    args = parser.parse_args()
    
    if args.id is None:
        args.id = list(range(args.start, args.end))
        args.id = [f'{i:04d}' for i in args.id]
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