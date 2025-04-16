import os
import cv2
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from termcolor import colored
from tqdm import tqdm
import zipfile
import shutil
import matplotlib
from demo.depth_anything_v2.dpt import DepthAnythingV2
from src.fmcw.simulator import Simulation
import matplotlib.pyplot as plt
from demo.video_depth_anything.video_depth import VideoDepthAnything
from demo.vitpose.vitpose_model import ViTPoseModel
from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate

def inference_batch_pose(model, images, device='cpu'):
    """
    Simplified batch inference function
    Args:
        model: ViTPose model
        images: List of numpy arrays [H,W,3] (BGR format)
        device: Computing device
    Returns:
        list[dict]: Keypoint prediction results for each image
    """
    cfg = model.cfg
    batch_data = []
    
    # Define flip pairs for keypoint symmetry
    flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                  [13, 14], [15, 16], [17, 20], [18, 21], [19, 22],
                  [23, 39], [24, 38], [25, 37], [26, 36], [27, 35], 
                  [28, 34], [29, 33], [30, 32], [40, 49], [41, 48], 
                  [42, 47], [43, 46], [44, 45], [54, 58], [55, 57], 
                  [59, 68], [60, 67], [61, 66], [62, 65], [63, 70], 
                  [64, 69], [71, 77], [72, 76], [73, 75], [78, 82], 
                  [79, 81], [83, 87], [84, 86], [88, 90], [91, 112], 
                  [92, 113], [93, 114], [94, 115], [95, 116], [96, 117], 
                  [97, 118], [98, 119], [99, 120], [100, 121], [101, 122], 
                  [102, 123], [103, 124], [104, 125], [105, 126], [106, 127], 
                  [107, 128], [108, 129], [109, 130], [110, 131], [111, 132]]
    
    # Process each image
    for img in images:
        h, w = img.shape[:2]
        center = np.array([w * 0.5, h * 0.5])
        
        # Calculate input size and scale
        input_size = cfg.data_cfg['image_size']
        aspect_ratio = input_size[0] / input_size[1]
        
        # Adjust dimensions based on aspect ratio
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25

        # Prepare single image data
        data = {
            'img': img,  
            'center': center,
            'scale': scale,
            'bbox_score': 1.0,
            'bbox_id': 0,
            'joints_3d': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation': 0,
            'flip_pairs': flip_pairs,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }
        
        data = Compose(cfg.test_pipeline)(data)
        batch_data.append(data)
    
    # Batch processing
    batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
    
    # Model inference
    with torch.no_grad():
        result = model(
            img=batch_data['img'].to(device),
            img_metas=batch_data['img_metas'].data[0],
            return_loss=False,
            return_heatmap=False)
    
    return result['preds']

def process_sequence(args, video_path, depth_estimator):
    """
    Process video sequence using batch inference
    Args:
        args: ArgumentParser object containing all parameters
        video_path: Path to the video file
        depth_estimator: Depth estimation model
    """
    if not os.path.exists(video_path):
        print(colored(f'Video file {video_path} does not exist', 'red'))
        return
    seq_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Initialize FMCW simulator
    simulator = Simulation(dtype=torch.float32, ctype=torch.complex64)
    simulator.simulator = simulator.simulator.to(args.device)
    
    # Initialize ViTPose model
    keypoint_detector = ViTPoseModel(args.device)
    print(colored('ViTPose model loaded', 'green'))
    
    # Read all frames
    print(colored('Reading video frames...', 'blue'))
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[20:-30, 20:-20]  # crop
        frames.append(frame)
    cap.release()
    print(colored(f'Read {len(frames)} frames', 'blue'))
    
    # Get keypoints using ViTPose
    print(colored('Detecting hand keypoints...', 'blue'))
    keypoints_list = []
    batch_size = args.batch_size
    
    # 批量处理关键点检测
    for i in tqdm(range(0, len(frames), batch_size)):
        batch_frames = frames[i:min(i + batch_size, len(frames))]
        preds = inference_batch_pose(keypoint_detector.model, batch_frames, args.device)
        
        for pred in preds:
            # 提取左右手关键点
            left_hand = pred[-42:-21]   # [21, 3]
            right_hand = pred[-21:]     # [21, 3]
            keypoints = np.concatenate([left_hand, right_hand], axis=0)  # [42, 3]
            keypoints_list.append(keypoints)
    
    keypoints = np.stack(keypoints_list)  # [N, 42, 3]
    keypoints = torch.from_numpy(keypoints).float().to(args.device)
    
    # Get depth maps
    print(colored('Estimating depth...', 'blue'))
    depths, _ = depth_estimator.infer_video_depth(
        frames, 30,  # assuming 30fps
        input_size=518,
        device=args.device,
        fp32=True,
        progress_bar=True
    )
    depths = torch.from_numpy(depths).float().to(args.device)  # [N, H, W]
    depths = depths - depths[:, depths.shape[1]//4, depths.shape[2]//2].mean()
    depths = depths / 10  # scale to proper size
    depths[depths < -0.5] -= 2.0  # background move further
    
    # Calculate 3D points for each keypoint
    print(colored('Computing 3D coordinates...', 'blue'))
    points_3d = []
    for frame_idx in range(len(frames)):
        kpts = keypoints[frame_idx]  # [42, 3]
        depth_map = depths[frame_idx]  # [H, W]
        
        # Get depth values at keypoint locations
        u = (kpts[:, 0] * depth_map.shape[1]).long().clamp(0, depth_map.shape[1]-1)
        v = (kpts[:, 1] * depth_map.shape[0]).long().clamp(0, depth_map.shape[0]-1)
        z = depth_map[v, u]  # [42]
        
        # Convert to 3D coordinates
        x = (u - simulator.simulator.cx) / simulator.simulator.fx * z
        y = (v - simulator.simulator.cy) / simulator.simulator.fy * z
        
        points = torch.stack([x, y, z], dim=-1)  # [42, 3]
        points_3d.append(points)
    
    points_3d = torch.stack(points_3d)  # [N, 42, 3]
    
    # Calculate velocities (3D)
    points_3d_diff = points_3d[1:] - points_3d[:-1]  # [N-1, 42, 3]
    velocities_3d = points_3d_diff * 30  # Scale by fps to get m/s
    points_3d = points_3d[:-1]
    
    # Downsample by taking every 3rd frame
    points_3d = points_3d[::3]  # [N/3, 42, 3]
    velocities_3d = velocities_3d[::3]  # [N/3, 42, 3]
    
    # Process radar signals
    print(colored('Generating radar signals...', 'blue'))
    radar_signals = []
    for i in tqdm(range(len(points_3d))):
        points = points_3d[i]  # [42, 3]
        velocities = velocities_3d[i]  # [42, 3]
        radar_signal = simulator.forward(points, velocities)
        radar_signals.append(radar_signal)
    radar_signals = torch.stack(radar_signals)  # [T, doppler, range]
    
    # Visualize results
    time_doppler_signal = radar_signals.sum(dim=2)  # [T, doppler]
    time_range_signal = radar_signals.sum(dim=1)  # [T, range]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(time_doppler_signal.cpu().numpy().T, aspect='auto')
    plt.title('Time-Doppler')
    plt.subplot(1, 2, 2)
    plt.imshow(time_range_signal.cpu().numpy().T, aspect='auto')
    plt.title('Time-Range')
    plt.tight_layout()
    plt.savefig(f'output/{seq_id}_radar.png')
    plt.close()

def process_archive(args, archive_id, depth_estimator):
    """
    Process a single archive: extract, process videos, then cleanup
    Args:
        args: ArgumentParser object containing all parameters
        archive_id: ID of the archive to process
    """
    base_path = '/root/autodl-tmp/datasets/csl-news'
    zip_path = os.path.join(base_path, f'archives/archive_{archive_id}.zip')
    extract_path = os.path.join(base_path, f'videos/archive_{archive_id}')

    if not os.path.exists(zip_path):
        print(colored(f'Archive file {zip_path} does not exist', 'red'))
        return False

    try:
        # Get all MP4 files
        mp4_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))
        mp4_files.sort()

        # Process each video with progress bar
        for video_path in tqdm(mp4_files, desc=f'Processing archive_{archive_id}'):
            # for now, we only process Common-Concerns videos
            if not (
                'Common-Concerns' in video_path or 
                'Dragon-TV' in video_path
            ):
                print(colored(f'Skipping video: {video_path}', 'yellow'))
                continue
            try:
                process_sequence(args, video_path, depth_estimator)
            except Exception as e:
                print(colored(f'Failed to process: {video_path}', 'red'))
                print(e)
                continue

    except Exception as e:
        print(colored(f'Error processing archive_{archive_id}: {str(e)}', 'red'))
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description='Depth Estimation for Video Sequences')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--id', nargs='+', default=None, help='List of sequences to process')
    parser.add_argument('--start', type=int, default=None, help='Start archive number')
    parser.add_argument('--end', type=int, default=None, help='End archive number')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitl'])
    args = parser.parse_args()
    
    # Process ID range if not explicitly provided
    if args.id is None:
        args.id = list(range(args.start, args.end + 1))
        args.id = [f'{i:03d}' for i in args.id]

    args.device = 'cuda'
    
    # Initialize VideoDepthAnything
    print(colored('Initializing depth estimation model...', 'blue'))
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    depth_estimator = VideoDepthAnything(**model_configs[args.encoder])
    depth_estimator.load_state_dict(
        torch.load(f'demo/video_depth_anything/checkpoints/video_depth_anything_{args.encoder}.pth', 
                  map_location='cpu', weights_only=True), 
        strict=True
    )
    depth_estimator = depth_estimator.to(args.device).eval()
    print(colored('Depth estimation model loaded successfully', 'green'))

    # Process each archive in range
    for archive_id in args.id:
        print(colored(f'\nProcessing archive_{archive_id}', 'blue'))
        print(colored('=' * 50, 'blue'))
        
        success = process_archive(args, archive_id, depth_estimator)
        
        if success:
            print(colored(f'Successfully completed archive_{archive_id}', 'green'))
        else:
            print(colored(f'Failed to process archive_{archive_id}', 'red'))
        
        print(colored('=' * 50, 'blue'))

    print(colored('\nAll archives processing completed', 'green'))

if __name__ == '__main__':
    main()