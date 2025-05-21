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
from scipy.ndimage import gaussian_filter1d


data_stats = {
    'csl-news/poses': {
        'mean': np.array([0.00681697, 0.11669894, -0.30188322]),
        'std': np.array([0.10176238, 0.11198849, 0.11414795]),
    },
    'csl-news/pred_poses': {
        'mean': np.array([]),
        'std': np.array([]),
    },
    'csl-daily/sentence/poses': {
        'mean': np.array([0.02832265, 0.37857532, -0.21782798]),
        'std': np.array([0.32808729, 0.3750018, 0.12171327]),
    },
    'csl-daily/sentence/pred_poses_0521': {
        'mean': np.array([0.00402528, 0.01333136, -0.01058482]),
        'std': np.array([0.10319765, 0.10940502, 0.08362551]),
    },
}

class BaseDataset(Dataset):
    """Base dataset class with common functionality"""
    def __init__(self, opt=None, split_path=None):
        self.opt = opt
        self.norm_pose = opt.get('norm_pose', False)
        self.max_length = opt.get('max_length', 256)
        
        # Load data paths
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)
            
    def __len__(self):
        return len(self.data_dict)
    
    def load_and_normalize_pose(self, pose_path):
        # load pose
        pose = np.load(pose_path)  # (T, 2, 24, 3)

        # normalize pose
        if self.norm_pose:
            mean, std = None, None
            for folder_name in data_stats.keys():
                if folder_name in pose_path:
                    mean = data_stats[folder_name]['mean']
                    std = data_stats[folder_name]['std']
                    break
            assert mean is not None and std is not None, f"Unknown pose path: {pose_path}"
            pose = (pose - mean) / std
        return pose
    
class SingleFrameDataset(BaseDataset):
    """Dataset for single frame pose/feature data"""
    
    def __init__(self, opt=None, split_path=None):
        super().__init__(opt, split_path)
        self.load_feature = opt.get('load_feature', False)
    
    def __getitem__(self, index):
        id, pose_path = list(self.data_dict.items())[index]
        pose = self.load_and_normalize_pose(pose_path) # (T, 2, 24, 3)
        pose = pose * 0.1 # scale to real-world scale

        # Randomly select a frame
        frame_idx = random.randint(0, min(pose.shape[0] - 2, self.max_length - 1))
        joints = pose[frame_idx]  # (2, 24, 3)

        if self.load_feature:
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

class SequenceBaseDataset(BaseDataset):
    """Base class for sequence-based datasets with common functionality"""
    
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
            
        # Load captions/annotations
        caption_path = opt.get('caption_path', 'dataset/CSL_News_Labels_converted.json')
        self.caption_dict = self._load_captions(caption_path)

    def pad_sequence(self, sequence, pad_shape):
        """Pad or truncate sequences to max_length"""
        valid_length = sequence.shape[0]
        if valid_length > self.max_length:
            # If sequence is longer than max_length, randomly sample a segment
            valid_indices = sorted(random.sample(range(valid_length), k=self.max_length))
            valid_length = self.max_length
            return sequence[valid_indices], valid_length
        else:
            pad_width = tuple([(0, self.max_length - valid_length)] + 
                            [(0, 0) for _ in range(len(pad_shape))])
            return np.pad(sequence, pad_width, mode='constant'), valid_length
    
    def load_features(self, pose_path):
        """Load pre-computed features"""
        feature_path = pose_path.replace('poses', self.feature_config['feature_dir'])
        return np.load(feature_path)  # (T, feature_dim)
    
    def load_pred_pose(self, pose_path):
        """Load predicted pose data"""
        pred_pose_path = pose_path.replace('poses', self.pose_config['pose_dir'])
        pose = self.load_and_normalize_pose(pred_pose_path) # (T, 2, 24, 3)
        if 'pred_poses' in pose_path:
            pose = gaussian_filter1d(pose, sigma=1.0, axis=0)
        return pose
    
    def load_raw_pose(self, pose_path): 
        """Load raw pose data and compute velocities"""
        pose = self.load_and_normalize_pose(pose_path) * 0.1 # (T, 2, 24, 3)
        velocities = (pose[1:] - pose[:-1]) * 30
        return pose[:-1], velocities
    
    def __getitem__(self, index):
        id, data_path = list(self.data_dict.items())[index]
        output_dict = {
            'id': id,
            'caption': self.caption_dict.get(id, "")
        }
                
        # Load features if enabled
        if self.modalities['use_features']:
            raise NotImplementedError("Features are not implemented.")
            features = self.load_features(data_path)
            features, valid_length = self.pad_sequence(features, features.shape[1:])
            output_dict['features'] = torch.from_numpy(features).float()
        
        # Load predicted poses if enabled
        if self.modalities['use_pred_pose']:
            pred_pose = self.load_pred_pose(data_path)
            pred_pose, valid_length = self.pad_sequence(pred_pose, pred_pose.shape[1:])
            output_dict['joints'] = torch.from_numpy(pred_pose).float()
        
        # Load raw pose data if enabled
        if self.modalities['use_raw_pose']:
            joints, velocities = self.load_raw_pose(data_path)
            joints, _ = self.pad_sequence(joints, joints.shape[1:])
            velocities, valid_length = self.pad_sequence(velocities, velocities.shape[1:])
            output_dict['joints'] = torch.from_numpy(joints).float()
            output_dict['velocities'] = torch.from_numpy(velocities).float()
        
        output_dict['valid_length'] = valid_length
        output_dict['path'] = data_path
        return output_dict

class CslNewsDataset(SequenceBaseDataset):
    """Dataset for CSL-News dataset"""
    def __init__(self, opt, split_path=None):
        super().__init__(opt, split_path)
    
    def _load_captions(self, caption_path):
        """Load caption/annotation data from file"""
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                return json.load(f)
        return {id: "" for id in self.data_dict.keys()}

class CslDailyDataset(SequenceBaseDataset):
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
