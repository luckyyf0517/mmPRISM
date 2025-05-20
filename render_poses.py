import os
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from make_video import make_video
from src.utils.plot import plot_hand_subplot
camera_params = {
    'camera_position': [0, 0, -2000],
    'camera_intrinsic': [1000, 1000, 256.0, 256.0]
}

if __name__ == '__main__':
    data_path = random.choice(glob('/root/autodl-tmp/datasets/csl-news/pred_poses/archive_*/*.npy'))
    # data_path = random.choice(glob('/root/autodl-tmp/datasets/csl-daily/sentence/pred_poses/*.npy'))
    print(data_path)

    data = np.load(data_path)

    # Apply Gaussian filter along the first dimension (time)
    data = gaussian_filter1d(data, sigma=1.0, axis=0)

    data_gt = np.load(data_path.replace('pred_poses', 'poses'))
    if 'daily' in data_path: 
        data_gt = data_gt * 0.5 + np.array([0.0, -0.2, 0.0])

    file_id = os.path.basename(data_path).split('.')[0]
    frame_idx = random.randint(0, data.shape[0] - 1)

    # Create output directory if it doesn't exist
    os.makedirs(f'outputs/{file_id}', exist_ok=True)

    # Plot and save each frame
    for i in tqdm(range(data.shape[0])):
        plt.figure(figsize=(12, 5))
        plot_hand_subplot(121, camera_params, data_gt[i, 0] * 1e3, data_gt[i, 1] * 1e3, 'Ground Truth')
        plot_hand_subplot(122, camera_params, data[i, 0] * 1e3, data[i, 1] * 1e3, 'Prediction')
        plt.savefig(f'outputs/{file_id}/frame_{i:04d}.png')
        plt.close()

    print(f'Saved to outputs/{file_id}')
    make_video(f'outputs/{file_id}', f'outputs/{file_id}.mp4', fps=30)