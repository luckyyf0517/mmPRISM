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

        
class mmSingleImageDataset(Dataset):
    def __init__(self, opt=None, split_path=None):
        self.opt = opt
        self.mode = opt.get('mode', 'pose')  # 'pose' or 'feature'
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
                np.concatenate([pose[[5,7,9], :], pose[-42:-21, :]], axis=0),
                np.concatenate([pose[[6,8,10], :], pose[-21:, :]], axis=0)
            ], axis=-3)
            
        frame_idx = random.randint(0, len(pose) - 2)
        points_3d = pose[frame_idx] # [17+21+21, 3]
        joints = process_pose(points_3d) # (2, 24, 3)
        
        if self.mode == 'feature':
            # Load pre-computed features
            feature_path = pose_path.replace('poses', 'features')
            features = np.load(feature_path)  # (T, feature_dim)
            features = features[frame_idx]  # (feature_dim)
            
            return {
                'id': id,
                'joints': joints,  # (2, 24, 3)
                'features': features,  # (feature_dim)
            }
        else:
            velocities_3d = (pose[frame_idx+1] - pose[frame_idx]) * 30
            return {
                'id': id, 
                'joints': joints, # (2, 24, 3)
                'points_3d': points_3d, # (57, 3)
                'velocities_3d': velocities_3d, # (57, 3)
            }

class mmWaveSequenceDataset(Dataset):
    """Dataset for millimeter wave time series data with captions"""
    
    def __init__(self, opt, split_path=None):
        self.opt = opt
        self.max_length = opt.get('max_length', 192)
        self.mode = opt.get('mode', 'pose')  # 'pose' or 'feature'
        
        # Load data paths
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)
            
        # Load captions if available
        caption_path = opt.get('caption_path', 'dataset/CSL_News_Labels_converted.json')
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                self.caption_dict = json.load(f)
        else:
            self.caption_dict = {id: "" for id in self.data_dict.keys()}

    def __len__(self):
        return len(self.data_dict)
    
    def load_pose_data(self, pose_path):
        """Mode 1: Load and process pose data"""
        # Load pose data
        pose = np.load(pose_path)  # (T, 59, 3)
        
        # Calculate velocities
        velocities_3d = (pose[1:] - pose[:-1]) * 30
        velocities_3d = np.concatenate([velocities_3d, velocities_3d[-1:]], axis=0)
        
        # Process joints
        def process_pose(pose):
            return np.stack([
                np.concatenate([pose[[5,7,9], :], pose[-42:-21, :]], axis=0),
                np.concatenate([pose[[6,8,10], :], pose[-21:, :]], axis=0)
            ], axis=-3)
        
        joints = np.array([process_pose(frame) for frame in pose])
        
        return pose, velocities_3d, joints
    
    def load_feature_data(self, pose_path):
        """Mode 2: Load pre-computed features"""
        feature_path = pose_path.replace('poses', 'features')
        features = np.load(feature_path)  # (T, feature_dim)
        return features

    def __getitem__(self, index):
        id, data_path = list(self.data_dict.items())[index]
        
        if self.mode == 'pose':
            # Mode 1: Load and process pose data
            points_3d, velocities_3d, joints = self.load_pose_data(data_path)
            valid_length = joints.shape[0]
            
            # Truncate or pad sequence
            if valid_length > self.max_length:
                joints = joints[:self.max_length]
                points_3d = points_3d[:self.max_length]
                velocities_3d = velocities_3d[:self.max_length]
            else:
                pad_width = ((0, self.max_length - joints.shape[0]), (0, 0), (0, 0), (0, 0))
                joints = np.pad(joints, pad_width, mode='constant')
                pad_width = ((0, self.max_length - points_3d.shape[0]), (0, 0), (0, 0))
                points_3d = np.pad(points_3d, pad_width, mode='constant')
                velocities_3d = np.pad(velocities_3d, pad_width, mode='constant')
            
            return {
                'id': id,
                'path': data_path, 
                'valid_length': valid_length,
                'joints': torch.from_numpy(joints).float(),
                'points_3d': torch.from_numpy(points_3d).float(),
                'velocities_3d': torch.from_numpy(velocities_3d).float(),
                'caption': self.caption_dict.get(id, "")
            }
            
        else:  # mode == 'feature'
            # Mode 2: Load pre-computed features
            features = self.load_feature_data(data_path)
            
            # Truncate or pad sequence
            if features.shape[0] > self.max_length:
                features = features[:self.max_length]
            else:
                pad_width = ((0, self.max_length - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')
            
            return {
                'id': id,
                'features': torch.from_numpy(features).float(),  # (T, feature_dim)
                'caption': self.caption_dict.get(id, "")
            }


if __name__ == '__main__':
    # Test mmWaveSequenceDataset in pose mode
    print("\nTesting mmWaveSequenceDataset in pose mode...")
    dataset_pose = mmWaveSequenceDataset({
        'max_length': 160,
        'mode': 'pose'
    }, split_path='dataset/csl-news-demo02/all.json')
    print(f"Dataset size: {len(dataset_pose)}")
    
    sample_pose = dataset_pose[0]
    print("\nSample data (pose mode):")
    print(f"ID: {sample_pose['id']}")
    print(f"Joints shape: {sample_pose['joints'].shape}")
    print(f"Points 3D shape: {sample_pose['points_3d'].shape}")
    print(f"Velocities 3D shape: {sample_pose['velocities_3d'].shape}")
    print(f"Caption: {sample_pose['caption']}")
    
    # Test DataLoader in pose mode
    dataloader_pose = DataLoader(dataset_pose, batch_size=2, shuffle=True)
    batch_pose = next(iter(dataloader_pose))
    print("\nBatch data shapes (pose mode):")
    print(f"Batch joints shape: {batch_pose['joints'].shape}")
    print(f"Batch points 3D shape: {batch_pose['points_3d'].shape}")
    print(f"Batch velocities 3D shape: {batch_pose['velocities_3d'].shape}")
    
    # Test mmWaveSequenceDataset in feature mode
    print("\nTesting mmWaveSequenceDataset in feature mode...")
    dataset_feature = mmWaveSequenceDataset({
        'max_length': 160,
        'mode': 'feature'
    }, split_path='dataset/csl-news-demo02/all.json')
    print(f"Dataset size: {len(dataset_feature)}")
    
    sample_feature = dataset_feature[0]
    print("\nSample data (feature mode):")
    print(f"ID: {sample_feature['id']}")
    print(f"Features shape: {sample_feature['features'].shape}")
    print(f"Caption: {sample_feature['caption']}")
    
    # Test DataLoader in feature mode
    dataloader_feature = DataLoader(dataset_feature, batch_size=2, shuffle=True)
    batch_feature = next(iter(dataloader_feature))
    print("\nBatch data shapes (feature mode):")
    print(f"Batch features shape: {batch_feature['features'].shape}")