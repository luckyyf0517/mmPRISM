import os
import cv2
import glob
import time
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class mmCSLNewsDataset(Dataset):
    def __init__(self, opt, split_path=None):
        self.opt = opt
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)

        self.max_length = opt.get('max_length', 192)
        with open('dataset/CSL_News_Labels_converted.json', 'r') as f:
            self.caption_dict = json.load(f)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        id, pose_path = list(self.data_dict.items())[index]
        pose = np.load(pose_path) # (T, 2, H, W)
        pose = np.ascontiguousarray(pose)
        if pose.shape[0] > self.max_length:
            pose = pose[:self.max_length]
        else:
            pose = np.pad(pose, ((0, self.max_length - pose.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant')
        caption = self.caption_dict[id]
        return {
            'id': id, 
            'pose': pose,
            'caption': caption,
        }
        
        
class mmSingleImageDataset(Dataset):
    def __init__(self, opt=None, split_path=None):
        self.opt = opt
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        id, pose_path = list(self.data_dict.items())[index]
        pose = np.load(pose_path) # (T, 59, 3)
        
        def process_pose(pose):
            """Extract arm and hand joints from pose data"""
            # Extract arm and hand joints
            return np.stack([
                np.concatenate([pose[[6,8,10], :], pose[-42:-21, :]], axis=0),
                np.concatenate([pose[[7,9,11], :], pose[-21:, :]], axis=0)
            ], axis=-3)
            
        frame_idx = random.randint(0, len(pose) - 2)
        points_3d = pose[frame_idx] # [17+21+21, 3]
        velocities_3d = (pose[frame_idx+1] - pose[frame_idx]) * 30
        joints = process_pose(points_3d) # (2, 24, 3)
        
        return {
            'id': id, 
            'joints': joints,
            'points_3d': points_3d,
            'velocities_3d': velocities_3d,
        }


if __name__ == '__main__':
    # Test mmSingleImageDataset
    print("\nTesting mmSingleImageDataset...")
    split_path = 'dataset/csl-news-demo02/train.json'
    dataset = mmSingleImageDataset({}, split_path)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print("\nSample data:")
    print(f"ID: {sample['id']}")
    print(f"Joints shape: {sample['joints'].shape}")
    print(f"Points 3D shape: {sample['points_3d'].shape}")
    print(f"Velocities 3D shape: {sample['velocities_3d'].shape}")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print("\nBatch data shapes:")
    print(f"Batch joints shape: {batch['joints'].shape}")
    print(f"Batch points 3D shape: {batch['points_3d'].shape}")
    print(f"Batch velocities 3D shape: {batch['velocities_3d'].shape}")
