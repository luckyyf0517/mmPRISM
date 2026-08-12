import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from scipy.ndimage import gaussian_filter1d
from make_video import make_video
from src.utils.plot import plot_hand_cv2
from src.data.dataset import CslDailyDataset, CollectedDailyDataset

camera_params = {
    'camera_position': [0, 0, -1000],
    'camera_intrinsic': [800, 800, 256.0, 256.0]
}

if __name__ == '__main__':
    # Configure dataset parameters
    opt = edict({
        "annotation_path": 'data/csl-daily/sentence_label/csl2020ct_v2.pkl',
        "max_length": 512,
        "modalities": {
            "use_pred_pose": True,
            "use_gt_pose": True,
            "use_raw_pose": False, 
            "use_features": False,
        },
        "pose_config": {
            "pose_dir": "pred_poses", 
            "norm_pose": True
        },
    })
    # Initialize dataset
    # dataset = CslDailyDataset(opt, split_path='dataset/csl-daily/all.json')
    dataset = CollectedDailyDataset(opt, split_path='dataset/tmp/test_demo.json')
    sample = dataset[0]

    length = sample['valid_length']
    
    # Get predicted poses and ground truth poses
    data_gt = sample['joints_gt'].numpy() * 0.1
    data_pred = sample['joints'].numpy() # * 0.1

    # Create time points for original and interpolated sequences
    t_orig = np.arange(length)
    t_interp = np.linspace(0, length-1, length*3)
    
    # Interpolate ground truth poses
    from scipy.interpolate import interp1d
    interp_func_gt = interp1d(t_orig, data_gt[:length], axis=0, kind='linear')
    data_gt = interp_func_gt(t_interp)
    
    # Interpolate predicted poses 
    interp_func_pred = interp1d(t_orig, data_pred[:length], axis=0, kind='linear')
    data_pred = interp_func_pred(t_interp)
    
    # Update sequence length after interpolation
    length = len(data_gt)
    
    # Get sequence ID
    file_id = sample['id']
    
    # Create output directory
    os.makedirs(f'outputs/{file_id}', exist_ok=True)

    # Plot and save each frame
    for i in tqdm(range(length - 1)):
        # Create a white canvas
        img = np.ones((512, 1024, 3), dtype=np.uint8) * 255
        
        # Draw ground truth hands
        img[:, :512] = plot_hand_cv2(camera_params, joints_l=data_gt[i, 0] * 1e3, joints_r=data_gt[i, 1] * 1e3)
        img[:, 512:] = plot_hand_cv2(camera_params, joints_l=data_pred[i, 0] * 1e3, joints_r=data_pred[i, 1] * 1e3)
        
        # Add titles
        cv2.putText(img, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, 'Prediction', (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save frame
        cv2.imwrite(f'outputs/{file_id}/frame_{i:04d}.png', img)

    print(f'Saved to outputs/{file_id}')
    make_video(f'outputs/{file_id}', f'outputs/{file_id}.mp4', fps=30)