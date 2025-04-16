import os
import sys
sys.path.append('demo/mmpose/projects/rtmpose3d')  

import cv2
import torch
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm
import mmcv
import mmengine
import matplotlib.pyplot as plt
from mmengine.logging import print_log
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.visualization import Pose3dLocalVisualizer
from src.fmcw.simulator import Simulation
from scipy.ndimage import gaussian_filter1d


def draw_2d_keypoints(frame, keypoints, color=(0, 255, 0), thickness=2):
    """Draw 2D hand keypoints and connections exactly as MMPose visualizer
    Args:
        frame: Input image (H, W, 3)
        keypoints: Hand keypoints with shape (21, 3)
        color: BGR color
        thickness: Line thickness
    """
    h, w = frame.shape[:2]
    
    # 定义手部骨架连接
    skeleton = [
        (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 手掌
        (0, 1), (1, 2), (2, 3), (3, 4),      # 大拇指
        (5, 6), (6, 7), (7, 8),              # 食指
        (9, 10), (10, 11), (11, 12),         # 中指
        (13, 14), (14, 15), (15, 16),        # 无名指
        (17, 18), (18, 19), (19, 20)         # 小指
    ]
    
    # 直接使用原始坐标
    points = keypoints.copy()
    points[:, 0] = points[:, 0]  # x坐标
    points[:, 1] = points[:, 1]  # y坐标
    points = points.astype(np.int32)
    
    # 绘制骨架连接
    for start_idx, end_idx in skeleton:
        pos1 = tuple(points[start_idx])
        pos2 = tuple(points[end_idx])
        
        # 检查点是否在图像范围内
        if (0 <= pos1[0] < w and 0 <= pos1[1] < h and 
            0 <= pos2[0] < w and 0 <= pos2[1] < h):
            cv2.line(frame, pos1, pos2, color, thickness)
    
    # 绘制关键点
    for point in points:
        if 0 <= point[0] < w and 0 <= point[1] < h:
            cv2.circle(frame, tuple(point), radius=3, color=color, thickness=-1)
    
    return frame


def process_single_image(frame, pose_estimator, visualizer, args):
    """Process single image using RTMPose3D
    """
    h, w = frame.shape[:2]
    bbox = np.array([[0, 0, w, h]])  # Single bbox covering the whole frame
    
    # Estimate 3D pose
    pose_results = inference_topdown(pose_estimator, frame, bbox)
    return pose_results[0]  # 只取第一个人的结果

def process_sequence(args, video_path, pose_estimator, visualizer: Pose3dLocalVisualizer, simulator: Simulation):
    """Process video sequence using RTMPose3D
    """
    if not os.path.exists(video_path):
        print(colored(f'Video file {video_path} does not exist', 'red'))
        return
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Read frames
    print(colored('Reading video frames...', 'blue'))
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
    print(colored(f'Read {len(frames)} frames', 'blue'))
    
    # Process each frame
    print(colored('Processing frames...', 'blue'))

    
    keypoints_all = []
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx].copy()
        pose_results = process_single_image(frame, pose_estimator, visualizer, args)
        keypoints = pose_results.pred_instances.keypoints[0]
        keypoints_all.append(keypoints)
        
        # keypoints_transformed = pose_results.pred_instances.transformed_keypoints[0]
        # left_hand = keypoints_transformed[-42:-21]   # Last 42 points are hands, first 21 is left hand
        # right_hand = keypoints_transformed[-21:]      # Last 21 points is right hand
        # # Draw left hand in blue, right hand in red
        # frame = draw_2d_keypoints(frame, left_hand, color=(255, 0, 0))  # Blue
        # frame = draw_2d_keypoints(frame, right_hand, color=(0, 0, 255)) # Red
        # # Save the visualized frame if needed
        # os.makedirs(f'output/{seq_id}', exist_ok=True)
        # cv2.imwrite(f'output/{seq_id}/{frame_idx:04d}.jpg', frame)
        
    keypoints_all = np.array(keypoints_all)
    
    depths_center = keypoints_all[:, [6, 7], 2].mean()
    keypoints_all[..., 2] = keypoints_all[..., 2] - depths_center
    keypoints_all = keypoints_all[:, -42:] # [N, 42, 3]
    
    # Apply 1D Gaussian filter along time dimension with sigma=2
    keypoints_all = gaussian_filter1d(keypoints_all, sigma=2, axis=0)
    
    velocities_all = (keypoints_all[1:] - keypoints_all[:-1]) * 30
    keypoints_all = keypoints_all[:-1]
    num_frames = len(keypoints_all)
    
    mmwave_cubes = []
    for frame_idx in range(num_frames):
        keypoints = torch.from_numpy(keypoints_all[frame_idx]).to(args.device)
        velocities = torch.from_numpy(velocities_all[frame_idx]).to(args.device)
        mmwave_cube = simulator.forward(keypoints, velocities)
        mmwave_cubes.append(mmwave_cube)    
    mmwave_cubes = torch.stack(mmwave_cubes, dim=0) # [num_frames, doppler, range, azimuth, elevation]
    
    signal = mmwave_cubes.clone()
    time_doppler_signal = signal.sum([2,3,4])
    time_azimuth_signal = signal.sum([1,2,4])
    time_elevation_signal = signal.sum([1,2,3])
    # time_doppler_signal = time_doppler_signal / torch.max(time_doppler_signal, dim=1, keepdim=True)[0]
    # time_azimuth_signal = time_azimuth_signal / torch.max(time_azimuth_signal, dim=1, keepdim=True)[0]
    # time_elevation_signal = time_elevation_signal / torch.max(time_elevation_signal, dim=1, keepdim=True)[0]
    
    # plt.figure(figsize=(8, 12))
    # plt.subplot(311)
    # plt.imshow(time_doppler_signal.cpu().numpy().T, aspect='auto')
    # plt.title('Time-Doppler')
    # plt.subplot(312)
    # plt.imshow(time_azimuth_signal.cpu().numpy().T, aspect='auto')
    # plt.title('Time-Azimuth')
    # plt.subplot(313)
    # plt.imshow(time_elevation_signal.cpu().numpy().T, aspect='auto')
    # plt.title('Time-Elevation') 
    # plt.tight_layout()
    # plt.savefig('output.png')
    # plt.close()
    
    time_signal = torch.stack([time_doppler_signal, time_azimuth_signal, time_elevation_signal], dim=-1)
    save_path = video_path.replace('.mp4', '.npy').replace('videos', 'signals')
    np.save(save_path, time_signal.cpu().numpy())

 
def process_archive(args, archive_id, pose_estimator, visualizer, simulator):
    """Process a single archive
    """
    base_path = '/root/autodl-tmp/datasets/csl-news'
    zip_path = os.path.join(base_path, f'archives/archive_{archive_id}.zip')
    extract_path = os.path.join(base_path, f'videos/archive_{archive_id}')

    if not os.path.exists(zip_path):
        print(colored(f'Archive file {zip_path} does not exist', 'red'))
        return False
        
    # Extract archive
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Get all MP4 files
    mp4_files = []
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(os.path.join(root, file))
    mp4_files.sort()
    
    # Process each video
    for video_path in tqdm(mp4_files, desc=f'Processing archive_{archive_id}'):
        if not ('Common-Concerns' in video_path or 'Dragon-TV' in video_path):
            print(colored(f'Skipping video: {video_path}', 'yellow'))
            continue
        process_sequence(args, video_path, pose_estimator, visualizer, simulator)
        
    # Remove extracted files
    import shutil
    shutil.rmtree(extract_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='RTMPose3D Body Pose Estimation')
    parser.add_argument('--id', nargs='+', default=None, help='List of sequences to process')
    parser.add_argument('--start', type=int, default=None, help='Start archive number')
    parser.add_argument('--end', type=int, default=None, help='End archive number')
    parser.add_argument('--gpu', type=int, dest='gpu_ids', help='GPU IDs')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument('--kpt-thr', type=float, default=0.3, help='Keypoint threshold')
    args = parser.parse_args()
    
    if args.id is None:
        args.id = list(range(args.start, args.end + 1))
        args.id = [f'{i:03d}' for i in args.id]

    # Initialize model
    print(colored('Initializing model...', 'blue'))
    pose_config = 'demo/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py'
    pose_checkpoint = 'demo/mmpose/projects/rtmpose3d/demo/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth'
    
    pose_estimator = init_model(pose_config, pose_checkpoint, device=args.device)
    print(colored('Model loaded successfully', 'green'))
    
    # Initialize visualizer
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)
    
    # Initialize simulator
    simulator = Simulation()
    simulator.simulator = simulator.simulator.to(args.device)
    
    # Process each archive
    failed_seq_list = []
    for archive_id in args.id:
        print(colored(f'\nProcessing archive_{archive_id}', 'blue'))
        print(colored('=' * 50, 'blue'))
        
        success = process_archive(args, archive_id, pose_estimator, visualizer, simulator)
        
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