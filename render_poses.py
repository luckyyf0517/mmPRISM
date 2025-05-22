import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from scipy.ndimage import gaussian_filter1d
from make_video import make_video
from src.utils.plot import plot_hand_subplot
from src.data.dataset import CslDailyDataset

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
            "use_features": False,
            "use_pred_pose": True,
            "use_raw_pose": False
        },
        "pose_config": {
            "pose_dir": "poses"
        },
        "norm_pose": True,
    })
    # Initialize dataset
    dataset_gt = CslDailyDataset(opt, split_path='dataset/csl-daily/all.json')

    opt_pred = deepcopy(opt)
    opt_pred.pose_config.pose_dir = 'pred_poses_0521'
    dataset_pred = CslDailyDataset(opt_pred, split_path='dataset/csl-daily/all.json')

    # Randomly select a sample
    # idx = random.randint(0, len(dataset_gt) - 1)
    idx = 2963
    sample_gt = dataset_gt[idx]
    sample_pred = dataset_pred[idx]
    assert sample_gt['id'] == sample_pred['id'], f"File ID mismatch: {sample_gt['id']} != {sample_pred['id']}"

    length = sample_pred['valid_length']
    
    # Get predicted poses and ground truth poses
    data_gt = sample_gt['joints'].numpy() * 0.1  # ground truth poses
    data_pred = sample_pred['joints'].numpy() * 0.1  # predicted poses

    file_id = sample_gt['id']
    
    # Create output directory
    os.makedirs(f'outputs/{file_id}', exist_ok=True)

    # Plot and save each frame
    for i in tqdm(range(length)):
        plt.figure(figsize=(12, 5))
        plot_hand_subplot(121, camera_params, data_gt[i, 0] * 1e3, data_gt[i, 1] * 1e3, 'Ground Truth')
        plot_hand_subplot(122, camera_params, data_pred[i, 0] * 1e3, data_pred[i, 1] * 1e3, 'Prediction')
        plt.savefig(f'outputs/{file_id}/frame_{i:04d}.png')
        plt.close()

    print(f'Saved to outputs/{file_id}')
    make_video(f'outputs/{file_id}', f'outputs/{file_id}.mp4', fps=30)