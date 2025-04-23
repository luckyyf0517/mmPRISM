import os
import sys
sys.path.append('demo/mmpose/projects/rtmpose3d')  

import torch
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm
import time
from src.fmcw.simulator import Simulation
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def process_point_cloud(data):
    """
    Process a point cloud data of shape [T, N, 3] and return a new point cloud of shape [T, N', 3].
    
    Args:
        data: numpy array of shape [T, N, 3] representing the point cloud data.
        
    Returns:
        numpy array of shape [T, N', 3] representing the processed point cloud.
    """
    # Extract body and hand points
    body = data[:, 5:11, :]
    handl = data[:, -42:-21, :]
    handr = data[:, -21:, :]

    # Define body skeleton
    body_skeleton = np.array([
        (0, 2), (2, 4), 
        (1, 3), (3, 5), 
        (0, 1)
    ])

    # Vectorized interpolation of points
    def interpolate_points_vectorized(p1, p2, num_points=3):
        t_values = np.linspace(0, 1, num_points + 2)[1:-1]
        t_values = t_values[np.newaxis, :, np.newaxis]  # Reshape t_values for broadcasting
        return p1[:, np.newaxis, :] + (p2 - p1)[:, np.newaxis, :] * t_values

    # Vectorized addition of body points and interpolated points
    interpolated_points_list = []
    for i, j in body_skeleton:
        interpolated_points = interpolate_points_vectorized(body[:, i], body[:, j])
        interpolated_points_list.append(interpolated_points)
    interpolated_points = np.concatenate(interpolated_points_list, axis=1)
    all_body_points = np.concatenate((body, interpolated_points, handl, handr), axis=1)
    all_body_points[..., 2] *= 0.6
    all_body_points[..., 1] 
    return all_body_points


def process_sequence(args, pose_path, simulator: Simulation):
    """Process single sequence using saved pose data
    """
    if not os.path.exists(pose_path):
        print(colored(f'Pose file {pose_path} does not exist', 'red'))
        return
    seq_id = os.path.splitext(os.path.basename(pose_path))[0]
    print(colored(f'\n>> Processing sequence: {seq_id}', 'cyan'))
    
    # Load keypoints
    print(colored('[1] Loading keypoints...', 'blue'))
    keypoints_all = np.load(pose_path)
    print(colored(f'    [OK] Loaded keypoints with shape {keypoints_all.shape}', 'green'))
    
    # Apply smoothing and calculate velocities
    print(colored('[2] Processing keypoints...', 'blue'))
    keypoints_all = process_point_cloud(keypoints_all) 
    # Print min/max values for each dimension
    print(colored('    Min/Max values for each dimension:', 'blue'))
    print(f'    X: min = {keypoints_all[..., 0].min():.3f}, max = {keypoints_all[..., 0].max():.3f}')
    print(f'    Y: min = {keypoints_all[..., 1].min():.3f}, max = {keypoints_all[..., 1].max():.3f}') 
    print(f'    Z: min = {keypoints_all[..., 2].min():.3f}, max = {keypoints_all[..., 2].max():.3f}')
    
    # Update keypoints_all with interpolated data
    keypoints_all = gaussian_filter1d(keypoints_all, sigma=1, axis=0)
    velocities_all = (keypoints_all[1:] - keypoints_all[:-1]) * 10 
    keypoints_all = keypoints_all[:-1]
    velocities_all = velocities_all[::3] # 30fps -> 10fps
    keypoints_all = keypoints_all[::3]
    num_frames = len(keypoints_all)
    print(colored('    [OK] Keypoint processing completed', 'green'))
    
    # Generate radar signals
    print(colored('[3] Generating radar signals...', 'blue'))
    mmwave_signals = []    
    for frame_idx in tqdm(range(num_frames), desc='    Simulating', ncols=80):
        keypoints = torch.from_numpy(keypoints_all[frame_idx]).to(args.device)
        velocities = torch.from_numpy(velocities_all[frame_idx]).to(args.device)
        signal = simulator.forward(keypoints, velocities).cpu().numpy()
        
        ra = signal.sum(0)
        da = signal.sum(1) 
        signal = np.stack([ra, da], axis=0)
        mmwave_signals.append(signal)
    mmwave_signals = np.stack(mmwave_signals, axis=0)
    print(colored(f'    [OK] Radar signals generated with shape {mmwave_signals.shape}', 'green'))
    
    # Save results
    print(colored('[4] Saving results...', 'blue'))
    save_path = pose_path.replace('poses', 'signals')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    np.save(save_path, mmwave_signals) # [T, 64, N]
    print(colored(f'    [OK] Results saved to: {save_path}', 'green'))
    print(colored('[OK] Sequence processing completed\n', 'green', attrs=['bold']))


def process_archive(args, archive_id, simulator):
    """Process a single archive
    """
    base_path = '/root/autodl-tmp/datasets/csl-news'
    pose_dir = os.path.join(base_path, f'poses/archive_{archive_id}')

    if not os.path.exists(pose_dir):
        print(colored(f'[X] Pose directory {pose_dir} does not exist', 'red'))
        return False
    
    # Get all NPY files
    npy_files = []
    for root, dirs, files in os.walk(pose_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    npy_files.sort()
    
    # Process each sequence
    total_sequences = len(npy_files)
    start_time = time.time()
    processed_sequences = 0
    
    print(colored(f'[1] Processing {total_sequences} sequences...', 'blue'))
    for i, pose_path in enumerate(npy_files):
        if not ('Common-Concerns' in pose_path or 'Dragon-TV' in pose_path):
            print(colored(f'    [SKIP] {os.path.basename(pose_path)}', 'yellow'))
            continue
            
        process_sequence(args, pose_path, simulator)
        processed_sequences += 1
        
        if processed_sequences >= 2:
            avg_time_per_seq = (time.time() - start_time) / processed_sequences
            remaining_seqs = total_sequences - (i + 1)
            est_time_remaining = (remaining_seqs * avg_time_per_seq) / 60
            print(colored(f'[INFO] Progress: {i+1}/{total_sequences}, ETA: {est_time_remaining:.1f}min', 'cyan'))
    
    return True


def main():
    parser = argparse.ArgumentParser(description='MMWave Signal Simulation from Poses')
    parser.add_argument('--id', nargs='+', default=None, help='List of archives to process')
    parser.add_argument('--start', type=int, default=None, help='Start archive number')
    parser.add_argument('--end', type=int, default=None, help='End archive number')
    parser.add_argument('--gpu', type=int, dest='gpu_id', help='GPU ID')
    args = parser.parse_args()
    
    if args.id is None:
        args.id = list(range(args.start, args.end))
        args.id = [f'{i:03d}' for i in args.id]
    args.device = 'cuda:' + str(args.gpu_id)    

    # Initialize simulator
    print(colored('Initializing simulator...', 'blue'))
    simulator = Simulation()
    simulator.simulator = simulator.simulator.to(args.device)
    print(colored('Simulator initialized successfully', 'green'))
    
    # Process each archive
    failed_archives = []
    for archive_id in args.id:
        print(colored(f'\nProcessing archive_{archive_id}', 'blue'))
        print(colored('=' * 50, 'blue'))
        
        success = process_archive(args, archive_id, simulator)
        
        if success:
            print(colored(f'Successfully completed archive_{archive_id}', 'green'))
        else:
            failed_archives.append(archive_id)
            print(colored(f'Failed to process archive_{archive_id}', 'red'))
        
        print(colored('=' * 50, 'blue'))

    if len(failed_archives) > 0:
        print(colored(f'Failed archives: {failed_archives}', 'red'))
    else:
        print(colored('\nAll archives processed successfully', 'green'))

if __name__ == '__main__':
    main()