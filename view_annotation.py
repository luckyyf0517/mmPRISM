import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from termcolor import colored
from src.utils.plot import plot_hand_cv2

def view_annotation(args):
    """View pose annotations and save visualization videos
    """
    # Set camera parameters
    camera_params = {
        'camera_position': [0, 0, -1000],
        'camera_intrinsic': [800, 800, 256.0, 256.0]
    }

    # Get pose file path
    pose_path = args.pose_dir
    seq_id = os.path.splitext(os.path.basename(pose_path))[0]
    
    print(colored(f'\n>> Processing sequence: {seq_id}', 'cyan'))

    # Define output video path
    video_path = os.path.join(args.output_dir, f'{seq_id}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Skip if video exists and not forcing regeneration
    if os.path.exists(video_path) and not args.force:
        print(colored(f'    [SKIP] {seq_id}: {video_path} already exists', 'yellow'))
        return

    # Load pose data
    keypoints_all = np.load(pose_path) * 3 # [N, 2, 24, 3]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (512, 512))

    # Generate visualization for each frame
    for frame_idx in tqdm(range(len(keypoints_all)), desc='    Generating video', ncols=80):
        joints_l = np.concatenate([
            keypoints_all[frame_idx, 0, :3],  # left arm 3 points
            keypoints_all[frame_idx, 0, 3:]   # left hand 21 points
        ])
        joints_r = np.concatenate([
            keypoints_all[frame_idx, 1, :3],  # right arm 3 points
            keypoints_all[frame_idx, 1, 3:]   # right hand 21 points
        ])
        
        # Plot hand
        frame_vis = plot_hand_cv2(
            camera_params=camera_params,
            joints_l=joints_l * 2.0e2,
            joints_r=joints_r * 2.0e2
        )
        
        # Write frame
        out.write(frame_vis)
    
    # Release video writer
    out.release()
    print(colored(f'    [OK] Video saved to: {video_path}', 'green'))

def main():
    parser = argparse.ArgumentParser(description='View pose annotations')
    parser.add_argument('--pose-dir', type=str, required=True, help='Path to pose npy file')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save output videos')
    parser.add_argument('--force', action='store_true', help='Force regenerate existing videos')
    args = parser.parse_args()

    view_annotation(args)

if __name__ == '__main__':
    main()
