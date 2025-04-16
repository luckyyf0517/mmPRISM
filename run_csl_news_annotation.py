import os
import cv2
import torch
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm
import zipfile
import shutil
import matplotlib
from demo.depth_anything_v2.dpt import DepthAnythingV2
from src.fmcw.simulator import Simulation


def process_sequence(args, video_path):
    """
    Process video sequence using batch inference
    Args:
        args: ArgumentParser object containing all parameters
        video_path: Path to the video file
    """
    if not os.path.exists(video_path):
        print(colored(f'Video file {video_path} does not exist', 'red'))
        return
    
    # Initialize FMCW simulator
    simulator = Simulation(dtype=torch.float32, ctype=torch.complex64)
    
    # Read all frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    depths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[15:-35, 22:-22] # crop
        frames.append(frame)
        # Estimate depth
        depth = args.depth_estimator.infer_image(frame, input_size=518)  # Use default input size
        depths.append(depth)
    cap.release()
        
    # Process in batches
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    num_frames = len(frames)
    
    # Convert depth sequence to tensor for batch processing
    depths = np.stack(depths, axis=0)  # [N, H, W]
    depths = torch.from_numpy(depths).float().to(args.device)
    
    # Set batch size
    batch_size = min(args.batch_size, num_frames)
    num_batches = (num_frames + batch_size - 1) // batch_size
    
    # Store radar signals
    radar_signals = []
    
    # Batch processing for FMCW simulation
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_frames)
        depth_batch = depths[start_idx:end_idx]  # [B, H, W]
        
        # Calculate path information
        path_info = simulator.forward(depth_batch)  # [3, B, num_rx, max_num_paths]
        
        # Generate radar frame
        radar_frame = simulator.get_raw_radar_frame(
            path_info, 
            TX=simulator.TX,
            save_cuda_memory=True
        )  # [B, num_chirps, num_rx, num_samples]
        
        radar_signals.append(radar_frame.cpu().numpy())
    
    # Concatenate results from all batches
    radar_signals = np.concatenate(radar_signals, axis=0)  # [N, num_chirps, num_rx, num_samples]
    
    # Set output directory
    output_dir = video_path.replace('.mp4', '').replace('csl-news', 'csl-news-data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save depth sequence and radar signals
    np.save(os.path.join(output_dir, 'depths.npy'), depths.cpu().numpy())
    np.save(os.path.join(output_dir, 'radar_signals.npy'), radar_signals)
    
    # Visualization part
    if True:
        # Set up output directory and video writer
        output_dir = '/root/autodl-tmp/omniHand/output'
        output_path = os.path.join(output_dir, os.path.basename(video_path))

        # Get video dimensions
        frame_height = frames[0].shape[0]
        frame_width = frames[0].shape[1]
        margin_width = 50
        output_width = frame_width * 2 + margin_width
        
        # Initialize video writer with same fps as input video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, frame_height))
        # Set up colormap for depth visualization
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        # Process and write each frame
        for frame, depth in zip(frames, depths):
            # Normalize and colorize depth
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            # Create white margin between frames
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            # Combine frames horizontally
            combined_frame = cv2.hconcat([frame, split_region, depth])
            out.write(combined_frame)
        out.release()


def process_archive(args, archive_id):
    """
    Process a single archive: extract, process videos, then cleanup
    Args:
        args: ArgumentParser object containing all parameters
        archive_id: ID of the archive to process
    """
    base_path = '/root/autodl-tmp/datasets/csl-news'
    zip_path = os.path.join(base_path, f'archive_{archive_id}.zip')
    extract_path = os.path.join(base_path, f'archive_{archive_id}')

    if not os.path.exists(zip_path):
        print(colored(f'Archive file {zip_path} does not exist', 'red'))
        return False

    try:
        # Extract archive
        print(colored(f'Extracting archive_{archive_id}...', 'blue'))
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Get all MP4 files
        mp4_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))
        mp4_files.sort()

        # Process each video with progress bar
        for video_path in tqdm(mp4_files, desc=f'Processing archive_{archive_id}'):
            try:
                process_sequence(args, video_path)
            except Exception as e:
                print(colored(f'Failed to process: {video_path}', 'red'))
                print(e)
                continue

    except Exception as e:
        print(colored(f'Error processing archive_{archive_id}: {str(e)}', 'red'))
        return False
    
    finally:
        # Cleanup
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)

    return True

def main():
    parser = argparse.ArgumentParser(description='Depth Estimation for Video Sequences')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--id', nargs='+', default=None, help='List of sequences to process')
    parser.add_argument('--start', type=int, default=None, help='Start archive number')
    parser.add_argument('--end', type=int, default=None, help='End archive number')
    parser.add_argument('--depth-model', type=str, default='vits', 
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Depth estimation model size')
    args = parser.parse_args()
    
    # Process ID range if not explicitly provided
    if args.id is None:
        args.id = list(range(args.start, args.end + 1))
        args.id = [f'{i:03d}' for i in args.id]

    # Add device to args
    args.device = torch.device('cuda')
    
    # Initialize depth estimator and add to args
    print(colored('Initializing depth estimation model...', 'blue'))
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    args.depth_estimator = DepthAnythingV2(**model_configs[args.depth_model])
    args.depth_estimator.load_state_dict(
        torch.load(f'demo/depth_anything_v2/checkpoints/depth_anything_v2_{args.depth_model}.pth', map_location='cpu'))
    args.depth_estimator = args.depth_estimator.to(args.device).eval()
    print(colored('Depth estimation model loaded successfully', 'green'))

    # Process each archive in range
    for archive_id in args.id:
        print(colored(f'\nProcessing archive_{archive_id}', 'blue'))
        print(colored('=' * 50, 'blue'))
        
        success = process_archive(args, archive_id)
        
        if success:
            print(colored(f'Successfully completed archive_{archive_id}', 'green'))
        else:
            print(colored(f'Failed to process archive_{archive_id}', 'red'))
        
        print(colored('=' * 50, 'blue'))

    print(colored('\nAll archives processing completed', 'green'))

if __name__ == '__main__':
    main()