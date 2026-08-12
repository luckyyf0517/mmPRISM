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
from easydict import EasyDict as edict

data_stats = {
    'csl-news/poses': {
        'mean': np.array([0.00681697, 0.11669894, -0.30188322]),
        'std': np.array([0.10176238, 0.11198849, 0.11414795]),
    },
    'csl-daily/sentence/poses': {
        'mean': np.array([0.02832265, 0.37857532, -0.21782798]),
        'std': np.array([0.32808729, 0.3750018, 0.12171327]),
    },
    'csl-daily/sentence/pred_poses_0602_rtm': {
        'mean': np.array([0.00491593, 0.01073456, -0.0108644]),
        'std': np.array([0.10114002, 0.10167552, 0.08443051]),
    },
    'collected_base/poses': {
        'mean': np.array([-0.05493037, -0.32715766, -0.17995592]),
        'std': np.array([0.5, 0.5, 0.5]),
    },
    'collected_demo/poses': {
        'mean': np.array([-0.32624024, -0.11011549, -0.28252611]),
        'std': np.array([0.5, 0.5, 0.5]),
    },
    'collected_demo/pred_poses': {
        'mean': np.array([0.0, 0.0, 0.0]),
        'std': np.array([1.0, 1.0, 1.0]),
    },
    'collected_csl/poses': {
        'mean': np.array([-0.19711676, -0.1889512, -0.22599903]),
        'std': np.array([0.32612011, 0.34238776, 0.11730845]),
    },
}


class BaseDataset(Dataset):
    """Base dataset class with common functionality"""
    def __init__(self, opt=None, split_path=None):
        self.opt = opt
        self.max_length = opt.get('max_length', 256)
        self.downsample_factor = opt.get('downsample_factor', 1)
        self.upsample_factor = opt.get('upsample_factor', 1)
        
        # Load data paths
        self.split_path = split_path
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)
            
            
    def __len__(self):
        return len(self.data_dict)
    
    def load_and_normalize_pose(self, pose_path, target_frames=None):

        # load pose
        pose = np.load(pose_path)  # (T, 2, 24, 3)
        pose = pose[::self.downsample_factor]

        if self.upsample_factor > 1.0:
            # Directly interpolate along time dimension
            from scipy.interpolate import interp1d
            # Original time points
            t = np.arange(pose.shape[0])
            # New time points after upsampling
            t_new = np.linspace(0, pose.shape[0]-1, pose.shape[0]*self.upsample_factor)
            # Interpolate along time dimension for all other dimensions
            f = interp1d(t, pose, axis=0, kind='linear')
            # Get interpolated sequence
            pose = f(t_new)

            if 'collected' in pose_path:
                pose = pose[3:]

        # sample frames if target_frames is specified
        if target_frames is not None and pose.shape[0] > target_frames:
            indices = np.linspace(0, pose.shape[0] - 1, target_frames, dtype=int)
            pose = pose[indices]

        # normalize pose
        if self.pose_config.get('norm_pose', False):
            mean, std = None, None
            for folder_name in data_stats.keys():
                if folder_name in pose_path:
                    mean = data_stats[folder_name]['mean']
                    std = data_stats[folder_name]['std']
                    break
            assert mean is not None and std is not None, f"Missing data stats for {pose_path}"
            pose = (pose - mean) / std
            
        # if 'train' in self.split_path:
        #     # Add random global scaling (same scaling for the whole sequence)
        #     global_scale = np.random.uniform(0.8, 1.2, (1, 1, 1, 3))  # shape: (1, 1, 1, 3)
        #     pose = pose * global_scale

        pose = pose - pose[:, :, [0], :].mean(1, keepdims=True) # remove global translation

        # Optionally remove depth information (z-coordinate)
        if self.pose_config.get('no_depth', False):
            pose[..., 2] = 0.0

        return pose
    

class SingleFrameDataset(BaseDataset):
    """Dataset for single frame pose/feature data"""
    
    def __init__(self, opt=None, split_path=None):
        super().__init__(opt, split_path)
        self.pose_config = edict(
            norm_pose=opt.get('norm_pose', False))
        
    def __getitem__(self, index):
        id, pose_path = list(self.data_dict.items())[index]
        pose = self.load_and_normalize_pose(pose_path, target_frames=None) # (T, 2, 24, 3)
        pose = pose * np.array([0.1, 0.1, 0.03]) # scale to real-world scale

        frame_idx = random.randint(0, min(pose.shape[0] - 2, self.max_length - 1))
        joints = pose[frame_idx]  # (2, 24, 3)
        velocities = (pose[frame_idx+1] - pose[frame_idx]) * 30

        return {
            'id': id, 
            'joints': joints.astype(np.float32),  # (2, 24, 3)
            'velocities': velocities.astype(np.float32),  # (2, 24, 3)
        }
    

class CollectedSingleFrameDataset(BaseDataset):
    """Dataset for collected single frame pose/feature data"""
    
    def __init__(self, opt=None, split_path=None):
        super().__init__(opt, split_path)
        self.pose_config = edict(
            norm_pose=opt.get('norm_pose', False))
        
        # Add temporal configuration
        self.use_temporal = opt.get('use_temporal', False)
        self.use_temporal_pose = opt.get('use_temporal_pose', False)
        self.num_temporal_frames = opt.get('num_temporal_frames', 5)  # Default to 5 frames
        self.num_frames = 99
        
        self.just_load_one_frame = opt.get('just_load_one_frame', False)
        
        # build data list
        self.data_list = []
        for seq_id in self.data_dict.keys():
            if not self.just_load_one_frame:
                for frame_id in range(self.num_frames):
                    self.data_list.append((seq_id, frame_id))
            else: 
                self.data_list.append((seq_id, 0))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, index):
        id, frame_idx = self.data_list[index]
        pose_path = self.data_dict[id]
        pose = self.load_and_normalize_pose(pose_path, target_frames=None) # (T, 2, 24, 3)
        pose = pose * np.array([0.1, 0.1, 0.03]) # scale to real-world scale

        if self.use_temporal:
            # For temporal processing, load multiple consecutive frames
            # Create a window of frames centered around the current frame
            start_frame = max(0, frame_idx - self.num_temporal_frames // 2)
            end_frame = min(pose.shape[0], start_frame + self.num_temporal_frames)
            start_frame = max(0, end_frame - self.num_temporal_frames)  # Adjust if needed
            
            # Load temporal sequence of mmwave data
            mmwave_sequence = []
            
            for t in range(start_frame, end_frame):
                # Load mmwave data for frame t
                mmwave_path = pose_path.replace('poses', 'mmwave').replace('.npy', f'/{t:04d}.npy')
                if os.path.exists(mmwave_path):
                    mmwave = np.load(mmwave_path) 
                    mmwave = mmwave[..., 0] + mmwave[..., 1] * 1j
                    mmwave_sequence.append(mmwave)
                else:
                    # If file doesn't exist, pad with zeros
                    mmwave_sequence.append(np.zeros((128, 86, 256), dtype=np.complex64))
            
            # Stack frames to create temporal dimension: [T, ...]
            if mmwave_sequence:
                mmwave_temporal = np.stack(mmwave_sequence, axis=0)  # [T, num_chirps, num_antenna, num_samples]
            else:
                # Fallback if no frames were loaded
                mmwave_temporal = np.zeros((self.num_temporal_frames, 128, 86, 256), dtype=np.complex64)
            
            # For joints, return temporal sequence if use_temporal_pose is True
            if self.use_temporal_pose:
                joints = pose[start_frame:end_frame]  # (T, 2, 24, 3)
                # Pad if needed to ensure exact num_temporal_frames
                if joints.shape[0] < self.num_temporal_frames:
                    pad_shape = (self.num_temporal_frames - joints.shape[0],) + joints.shape[1:]
                    joints = np.concatenate([joints, np.zeros(pad_shape, dtype=joints.dtype)], axis=0)
            else:
                # For joints, we still return a single frame as the target
                joints = pose[frame_idx]  # (2, 24, 3)
            
            return {
                'id': id, 
                'joints': joints.astype(np.float32),  # (T, 2, 24, 3) if use_temporal_pose else (2, 24, 3)
                'mmwave': mmwave_temporal.astype(np.complex64),  # [T, num_chirps, num_antenna, num_samples] - temporal for TVAN
                'frame_idx': frame_idx,
            }
        else:
            # Original single frame behavior
            joints = pose[frame_idx]  # (2, 24, 3)
            
            mmwave_path = pose_path.replace('poses', 'mmwave').replace('.npy', f'/{frame_idx:04d}.npy')
            mmwave = np.load(mmwave_path) 
            mmwave = mmwave[..., 0] + mmwave[..., 1] * 1j
            
        # # add random noise to mmwave
        # mmwave = mmwave + np.random.normal(0, 0.01, mmwave.shape, dtype=np.complex64)

        return {
            'id': id, 
            'joints': joints.astype(np.float32),  # (2, 24, 3)
            'mmwave': mmwave.astype(np.complex64), 
            'frame_idx': frame_idx,
        }

class SequenceBaseDataset(BaseDataset):
    """Base class for sequence-based datasets with common functionality"""
    
    def __init__(self, opt, split_path=None):
        super().__init__(opt, split_path)
        
        # Load modality configuration
        self.modalities = opt.get('modalities', {
            'use_pred_pose': False,
            'use_gt_pose': False,
            'use_raw_pose': False, 
            'use_features': False,
        })
        self.feature_config = opt.get('feature_config', {
            'feature_dim': 512,
            'feature_dir': 'features'
        })
        self.pose_config = opt.get('pose_config', {
            'pose_dir': 'pred_poses', 
            'norm_pose': False,
            'no_depth': False,  # Option to remove depth information (z-coordinate)
        })
        
        # Validate modality settings
        if not any(self.modalities.values()):
            raise ValueError("At least one modality must be enabled")
            
        # Load captions/annotations
        caption_path = opt.get('caption_path', 'dataset/CSL_News_Labels_converted.json')
        self.caption_dict = self._load_captions(caption_path)

    def pad_sequence(self, sequence, pad_shape, valid_indices=None):
        """Pad or truncate sequences to max_length"""
        valid_length = sequence.shape[0]
        if valid_length > self.max_length:
            # Generate new random indices if not provided
            if valid_indices is None:
                valid_indices = sorted(random.sample(range(valid_length), k=self.max_length))
            valid_length = self.max_length
            return sequence[valid_indices], valid_length, valid_indices
        else:
            pad_width = tuple([(0, self.max_length - valid_length)] + 
                            [(0, 0) for _ in range(len(pad_shape))])
            return np.pad(sequence, pad_width, mode='constant'), valid_length, None
    
    def load_features(self, pose_path):
        """Load pre-computed features"""
        feature_path = pose_path.replace('poses', self.feature_config['feature_dir'])
        return np.load(feature_path)  # (T, feature_dim)
    
    def load_pred_pose(self, pose_path):
        """Load predicted pose data"""
        pred_pose_path = pose_path.replace('poses', self.pose_config['pose_dir'])
        pose = self.load_and_normalize_pose(pred_pose_path, target_frames=None) # (T, 2, 24, 3)
        if 'pred_poses' in pred_pose_path:
            pose = gaussian_filter1d(pose, sigma=1.0, axis=0)
        return pose

    def load_gt_pose(self, pose_path):
        """Load gt pose data"""
        pose = self.load_and_normalize_pose(pose_path, target_frames=None) # (T, 2, 24, 3)
        return pose
    
    def load_raw_pose(self, pose_path): 
        """Load raw pose data and compute velocities"""
        pose = self.load_and_normalize_pose(pose_path, target_frames=None) * 0.1 # (T, 2, 24, 3)
        velocities = (pose[1:] - pose[:-1]) * 30
        return pose[:-1], velocities
    
    def __getitem__(self, index):
        id, data_path = list(self.data_dict.items())[index]
        output_dict = {
            'id': id,
            'caption': self.caption_dict.get(id, "")
        }
        
        # Store sampling indices shared across all modalities
        shared_valid_indices = None
        valid_length = None
                
        # Load features if enabled
        if self.modalities['use_features']:
            features = self.load_features(data_path)
            features, valid_length, shared_valid_indices = self.pad_sequence(
                features, features.shape[1:], shared_valid_indices)
            output_dict['features'] = torch.from_numpy(features).float()
        
        # Load predicted poses if enabled
        if self.modalities['use_pred_pose']:
            pred_pose = self.load_pred_pose(data_path)
            pred_pose, valid_length, shared_valid_indices = self.pad_sequence(
                pred_pose, pred_pose.shape[1:], shared_valid_indices)
            output_dict['joints'] = torch.from_numpy(pred_pose).float()

        # Load gt poses if enabled
        if self.modalities['use_gt_pose']:
            gt_pose = self.load_gt_pose(data_path)
            gt_pose, valid_length, shared_valid_indices = self.pad_sequence(
                gt_pose, gt_pose.shape[1:], shared_valid_indices)
            output_dict['joints_gt'] = torch.from_numpy(gt_pose).float()
        
        # Load raw pose data if enabled
        if self.modalities['use_raw_pose']:
            joints, velocities = self.load_raw_pose(data_path)
            joints, _, shared_valid_indices = self.pad_sequence(
                joints, joints.shape[1:], shared_valid_indices)
            velocities, valid_length, _ = self.pad_sequence(
                velocities, velocities.shape[1:], shared_valid_indices)
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
        if caption_path is None or not os.path.exists(caption_path):
            # raise FileNotFoundError(f"Annotation file not found: {caption_path}")
            return {}
            
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
    

class CollectedDailyDataset(CslDailyDataset):
    def __init__(self, opt, split_path=None):
        super().__init__(opt, split_path)
    
    def __getitem__(self, index):
        output_dict = super().__getitem__(index)
        
        # Load mmwave data for the current frame (no temporal support)
        if self.modalities.get('use_mmwave', False):
            valid_length = output_dict['valid_length']
            frame_idx = min(valid_length - 1, 0)  # Use first frame
            mmwave_path = output_dict['path'].replace('poses', 'mmwave').replace('.npy', f'/{frame_idx:04d}.npy')
            if os.path.exists(mmwave_path):
                mmwave = np.load(mmwave_path) 
                mmwave = mmwave[..., 0] + mmwave[..., 1] * 1j
                output_dict['mmwave'] = mmwave
            else:
                # Fallback if file doesn't exist
                output_dict['mmwave'] = np.zeros((128, 86, 256), dtype=np.complex64)

        # Scale to real-world scale
        output_dict['joints'] = output_dict['joints'] * torch.tensor([0.1, 0.1, 0.03])
        
        return output_dict
    
