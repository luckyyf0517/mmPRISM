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

class BaseDataset(Dataset):
    """Base dataset class with common functionality"""
    def __init__(self, opt=None, split_path=None):
        self.opt = opt
        self.max_length = opt.get('max_length', 192)
        
        # Load data paths
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)
            
    def __len__(self):
        return len(self.data_dict)
    
    def pad_sequence(self, sequence, valid_length, pad_shape):
        """Pad or truncate sequences to max_length"""
        if valid_length > self.max_length:
            return sequence[:self.max_length]
        else:
            pad_width = tuple([(0, self.max_length - valid_length)] + 
                            [(0, 0) for _ in range(len(pad_shape))])
            return np.pad(sequence, pad_width, mode='constant')

class mmSingleImageDataset(BaseDataset):
    """Dataset for single frame pose/feature data"""
    
    def __init__(self, opt=None, split_path=None):
        super().__init__(opt, split_path)
        self.mode = opt.get('mode', 'pose')  # 'pose' or 'feature'
    
    def __getitem__(self, index):
        id, pose_path = list(self.data_dict.items())[index]
        pose = np.load(pose_path)  # (T, 2, 24, 3)
        
        # Randomly select a frame
        frame_idx = random.randint(0, min(pose.shape[0] - 2, self.max_length - 1))
        joints = pose[frame_idx]  # (2, 24, 3)

        # todo: scale
        if 'Daily' in pose_path:
            joints = joints * 0.5 + np.array([0.0, -0.2, 0.0])
        
        if self.mode == 'feature':
            # Load pre-computed features
            feature_path = pose_path.replace('poses', 'features')
            features = np.load(feature_path)[frame_idx]  # (feature_dim)
            
            return {
                'id': id,
                'joints': joints,  # (2, 24, 3)
                'features': features,  # (feature_dim)
            }
        else:
            velocities = (pose[frame_idx+1] - pose[frame_idx]) * 30
            return {
                'id': id, 
                'joints': joints,  # (2, 24, 3)
                'velocities': velocities,  # (2, 24, 3)
            }

class CslNewsDataset(BaseDataset):
    """Dataset for millimeter wave time series data with captions"""
    
    def __init__(self, opt, split_path=None):
        super().__init__(opt, split_path)
        
        # Load modality configuration
        self.modalities = opt.get('modalities', {
            'use_features': False,
            'use_pred_pose': False,
            'use_raw_pose': False,
        })
        self.feature_config = opt.get('feature_config', {
            'feature_dim': 512,
            'feature_dir': 'features'
        })
        self.pose_config = opt.get('pose_config', {
            'pose_dir': 'pred_poses'
        })
        
        # Validate modality settings
        if not any(self.modalities.values()):
            raise ValueError("At least one modality must be enabled")
            
        # Load captions if available
        caption_path = opt.get('caption_path', 'dataset/CSL_News_Labels_converted.json')
        self.caption_dict = self._load_captions(caption_path)
    
    def _load_captions(self, caption_path):
        """Load caption data from file"""
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                return json.load(f)
        return {id: "" for id in self.data_dict.keys()}
    
    def load_features(self, pose_path):
        """Load pre-computed features"""
        feature_path = pose_path.replace('poses', self.feature_config['feature_dir'])
        return np.load(feature_path)  # (T, feature_dim)
    
    def load_pred_pose(self, pose_path):
        """Load predicted pose data"""
        pred_pose_path = pose_path.replace('poses', self.pose_config['pose_dir'])
        return np.load(pred_pose_path)  # (T, 2, 24, 3)
    
    def load_raw_pose(self, pose_path):
        """Load raw pose data and compute velocities"""
        pose = np.load(pose_path)  # (T, 2, 24, 3)
        velocities = (pose[1:] - pose[:-1]) * 30

        # todo: scale 
        if 'Daily' in pose_path:
            pose = pose * 0.5 + np.array([0.0, -0.2, 0.0])
            velocities = velocities * 0.5

        return pose[:-1], velocities
    
    def __getitem__(self, index):
        id, data_path = list(self.data_dict.items())[index]
        output_dict = {
            'id': id,
            'caption': self.caption_dict.get(id, "")
        }
        
        valid_length = self.max_length
        
        # Load features if enabled
        if self.modalities['use_features']:
            features = self.load_features(data_path)
            valid_length = min(valid_length, features.shape[0])
            features = self.pad_sequence(features, valid_length, features.shape[1:])
            output_dict['features'] = torch.from_numpy(features).float()
        
        # Load predicted poses if enabled
        if self.modalities['use_pred_pose']:
            pred_pose = self.load_pred_pose(data_path)
            valid_length = min(valid_length, pred_pose.shape[0])
            pred_pose = self.pad_sequence(pred_pose, valid_length, pred_pose.shape[1:])
            output_dict['joints'] = torch.from_numpy(pred_pose).float()
        
        # Load raw pose data if enabled
        if self.modalities['use_raw_pose']:
            joints, velocities = self.load_raw_pose(data_path)
            valid_length = min(valid_length, joints.shape[0])
            joints = self.pad_sequence(joints, valid_length, joints.shape[1:])
            velocities = self.pad_sequence(velocities, valid_length, velocities.shape[1:])
            output_dict['joints'] = torch.from_numpy(joints).float()
            output_dict['velocities'] = torch.from_numpy(velocities).float()
        
        output_dict['valid_length'] = valid_length
        output_dict['path'] = data_path
        return output_dict

class CslDailyDataset(CslNewsDataset):
    """Dataset for CSL-Daily dataset with sentence annotations"""
    
    def __init__(self, opt, split_path=None):
        # Change default caption_path to annotation_path
        if 'annotation_path' in opt:
            opt['caption_path'] = opt['annotation_path']
        super().__init__(opt, split_path)
    
    def _load_captions(self, caption_path):
        """Override _load_captions to load sentence annotations from pickle file"""
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f"Annotation file not found: {caption_path}")
            
        import pickle
        with open(caption_path, 'rb') as f:
            data = pickle.load(f)
            
        # Create mapping from video ID to annotation
        caption_dict = {}
        for item in data['info']:
            # Construct video ID: name_signer_time
            video_id = item['name']
            # Use character-level annotation as caption
            caption = ''.join(item['label_char'])
            caption_dict[video_id] = caption
            
        return caption_dict

if __name__ == '__main__':
    # Test configuration
    opt = {
        'max_length': 192,
        'modalities': {
            'use_features': False,
            'use_pred_pose': False,
            'use_raw_pose': True,
        },
        'annotation_path': 'data/CSL-Daily/sentence_label/csl2020ct_v2.pkl',
    }
    
    # Initialize dataset
    dataset = CslDailyDataset(opt, split_path='dataset/csl-daily-demo01/val.json')
    print(f"Dataset size: {len(dataset)}")
    
    # Find maximum valid length in the dataset
    max_length = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample['caption'])