import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

import os
import time
import json
import einops
import numpy as np
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from einops.layers.torch import Rearrange
from tqdm import tqdm
from copy import deepcopy

from src.utils.tools import get_obj_from_str, instantiate_from_config
from src.fmcw.simulator import Simulation, Processor


class OmniHand(LightningModule):
    """Base model class with common functionality for all hand reconstruction models."""
    
    def __init__(self, cfg=None):
        super().__init__()
        # Tell lightning to not handle the optimization
        self.automatic_optimization = False
        
        self.save_hyperparameters()
        self.cfg = cfg
        # Handle batch_size from training config or top level
        self.batch_size = cfg.training.get('batch_size', cfg.get('batch_size', 8))
        
        # Get antenna array size configuration (default to "full")
        self.antenna_array_size = cfg.get('antenna_array_size', 'full')
        
        self.processor = Processor(learnable_weights=cfg.get('learnable_weights', False), array_size=self.antenna_array_size)
        self.use_simulator = cfg.get('use_simulator', True)
        if self.use_simulator:
            self.simulator = Simulation(array_size=self.antenna_array_size)
            self.processor.if_process_range = True
        
        # Initialize backbone (Vision Transformer)
        self.backbone = instantiate_from_config(cfg.backbone)
        if cfg.backbone.pretrained is not None:
            self.backbone.load_state_dict(torch.load(
                os.path.join(cfg.backbone.pretrained, 'model.pth'), weights_only=True), strict=True)
        if cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from backbone params (support CubeNet, CSPEncoder3D, MMHandEncoder, and TVAN)
        if hasattr(cfg.backbone.params, 'hidden_dims'):
            self.feature_dim = cfg.backbone.params.hidden_dims[-1]  # For CubeNet and MMHandEncoder
        elif hasattr(cfg.backbone.params, 'stage_channels'):
            self.feature_dim = cfg.backbone.params.stage_channels[-1]  # For CSPEncoder3D
        else:
            raise ValueError("Backbone params must have either 'hidden_dims' or 'stage_channels'")
        
        self.relu = nn.ReLU()
        
        # Handle Sequential regressor with custom layer building
        if cfg.regressor.target == 'torch.nn.Sequential' and hasattr(cfg.regressor.params, 'layers'):
            # Build Sequential module manually from layers configuration
            layers = []
            for layer_config in cfg.regressor.params.layers:
                layer_class = get_obj_from_str(layer_config.target)
                layer_instance = layer_class(**layer_config.params)
                layers.append(layer_instance)
            self.regressor = nn.Sequential(*layers)
        else:
            # Use standard instantiation
            self.regressor = instantiate_from_config(cfg.regressor)
            
            
        # Training parameters
        self.lr = cfg.training.lr
        self.betas = cfg.training.betas
        self.weight_decay = cfg.training.weight_decay
        
    def training_step(self, batch, batch_idx):
        # Forward pass once
        results = self.forward(batch)
        
        # Calculate reconstruction loss
        rec_loss_dict = self.compute_loss(results, batch)
        
        # Train generator
        g_opt = self.optimizers()
        g_opt.zero_grad()
        self.manual_backward(rec_loss_dict['loss'])
        g_opt.step()
            
        # Logging
        self._log_info(rec_loss_dict, phase='train')
        self._log_progress(batch_idx, rec_loss_dict)
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        results = self.forward(batch)  # [B, 2, 24, 3]
        # Compute losses
        loss_dict = self.compute_loss(results, batch)
        # Logging
        self._log_info(loss_dict, phase='valid')
        self._log_progress(batch_idx, loss_dict)
        return loss_dict['loss']  
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        results = self.forward(batch)  # [B, 2, 24, 3]
        # Compute losses
        loss_dict = self.compute_loss(results, batch)
        
        # Calculate 3DPCK@40mm
        pred = results['joints'] * 1e3  # [B, 2, 24, 3]
        target = batch['joints'] * 1e3  # [B, 2, 24, 3]
        
        # Only use palm joints (last 21 joints)
        pred_palm = pred[..., 3:, :]  # [B, 2, 21, 3]
        target_palm = target[..., 3:, :]  # [B, 2, 21, 3]
        
        # Get wrist positions for normalization
        pred_wrist = pred_palm.mean(dim=-2, keepdim=True)  # [B, 2, 1, 3]
        target_wrist = target_palm.mean(dim=-2, keepdim=True)  # [B, 2, 1, 3]
        
        # Normalize by subtracting wrist position
        pred_norm = pred_palm - pred_wrist  # [B, 2, 21, 3]
        target_norm = target_palm - target_wrist  # [B, 2, 21, 3]

        # Check if any value in the target is NaN
        valid_mask = ~torch.any(torch.isnan(target_norm), dim=-1)  # [B, 2, 21]
        
        if valid_mask.sum() > 0:
            # Calculate distances for valid joints
            distances = torch.norm(pred_norm[valid_mask] - target_norm[valid_mask], dim=-1)  # [N]
            pck_results = (distances <= 40.0).float()  # [N] - 1 if within threshold, 0 otherwise
            loss_dict['3DPCK@40mm'] = pck_results.mean()  # Average across all valid joints
        
        # Logging
        self._log_info(loss_dict, phase='test')
        self._log_progress(batch_idx, loss_dict)
        return loss_dict['loss']
    
    def forward(self, input_data):
        """Forward pass
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Dictionary containing either 'joints' or 'simcc_logits'
        """
        features = self.encode_feature(input_data)
        joints = self.forward_feature(features)
        return {
            'joints': joints,
        }
        
    def encode_feature(self, input_data):
        """Encode input data into features"""
        if self.use_simulator:
            joints = input_data['joints']
            velocities = input_data['velocities']
            x = self.simulator(joints, velocities)
        else:
            x = input_data['mmwave']
            
        # Check if we have temporal data (5D or 6D tensor: [B, T, ...])
        if (x.dim() == 5 or x.dim() == 6) and hasattr(self.backbone, 'temporal_frames'):
            # Temporal processing: check if data is already processed or raw
            B, T = x.shape[:2]
            
            # Check if this is already processed data for TVAN ([B, T, 64, 32, 32, 32])
            # or raw mmwave data ([B, T, num_chirps, num_antenna, num_samples])
            if x.dim() == 6 and x.shape[2] == 64 and x.shape[3] == 32 and x.shape[4] == 32 and x.shape[5] == 32:
                # Already processed data for TVAN - bypass processor
                x_temporal = x  # [B, T, 64, 32, 32, 32]
            else:
                # Raw mmwave data - process through Processor first
                # x is [B, T, num_chirps, num_antenna, num_samples]
                processed_frames = []
                
                for t in range(T):
                    # Process individual frame: [B, num_chirps, num_antenna, num_samples]
                    frame = x[:, t, :, :, :]
                    processed_frame = self.processor(frame)  # [B, 64, 32, 32, 32]
                    processed_frames.append(processed_frame.unsqueeze(1))  # Add time dimension
                
                # Stack processed frames: [B, T, 64, 32, 32, 32]
                x_temporal = torch.cat(processed_frames, dim=1)
            
            # Extract features using temporal backbone
            features = self.backbone(x_temporal)
            return features
        else:
            # Standard processing: x is [B, num_chirps, num_antenna, num_samples]
            x = self.processor(x)
    
        # Extract features using backbone
        features = self.backbone(x)
        return features
    
    def forward_feature(self, features):
        """Forward pass for feature extraction"""
        features = self.relu(features)
        regressor_output = self.regressor(features)
        
        B = features.shape[0]
        
        # Reshape for joints
        # Regressor outputs [B, 288] where we need first 144 for joints
        joint_features = regressor_output[:, :144]  # [B, 144] 
        reshaped_regressor = joint_features.view(B, 2, 24, 3)  # [B, 2, 24, 3]
        
        return reshaped_regressor
            
    def compute_loss(self, results, batch):
        """Calculate reconstruction losses
        Args:
            results: Model output containing either 'joints' or 'simcc_logits'
            batch: Ground truth data
        Returns:
            Dictionary containing reconstruction losses
        """
        loss_dict = {}
        
        # Traditional decoder: coordinate loss
        loss_dict = self._compute_coordinate_loss(results, batch)
        
        return loss_dict
    
    def _compute_coordinate_loss(self, results, batch):
        """Calculate coordinate losses for traditional decoder
        Args:
            results: Contains 'joints' [B, 2, 24, 3]
            batch: Ground truth data
        Returns:
            Dictionary containing coordinate losses
        """
        loss_dict = {}
        pred = results['joints'] * 1e3  # [B, 2, 24, 3]
        target = batch['joints'] * 1e3  # [B, 2, 24, 3]

        # Merge batch and hand dimensions for full joint set
        B = pred.shape[0]
        pred_full = pred.reshape(-1, 24, 3)  # [B*2, 24, 3]
        target_full = target.reshape(-1, 24, 3)

        # Check if any value in the last dimension (xyz) is NaN
        valid_mask = ~torch.any(torch.isnan(target_full), dim=-1)  # [B*2, 24]

        # Apply mask to both predictions and targets
        pred_valid = pred_full[valid_mask]  # [N, 3]
        target_valid = target_full[valid_mask]  # [N, 3]
        
        # L1 loss on valid joint positions (using all 24 joints)
        loss_dict['loss_joints'] = F.l1_loss(pred_valid, target_valid)
        
        # MPJPE (Mean Per Joint Position Error) on valid joints (using all 24 joints)
        mpjpe = torch.norm(pred_valid - target_valid, dim=-1) # [N]
        loss_dict['MPJPE'] = mpjpe.mean()  
        
        loss_dict['loss'] = loss_dict['loss_joints']
        return loss_dict
        

    def _log_info(self, info_dict, phase='train'):
        """Log losses to logger"""
        for k, v in info_dict.items():
            self.log(
                phase + '/' + k, v,
                on_step=self.training,
                on_epoch=not self.training,
                logger=True,
                batch_size=self.batch_size,
                sync_dist=True)
    
    def _log_progress(self, batch_idx, loss_dict):
        """Log training progress"""
        self.print(f'Epoch: {self.current_epoch:04d}, '
                  f'iter: {batch_idx:06d}, '
                  f'loss: {loss_dict["loss"].item():.4f}, '
                  f'MPJPE: {loss_dict["MPJPE"].item():.4f}')

    def configure_optimizers(self):
        """Configure optimizers for generator"""
        # Generator optimizer (backbone + regressor)
        g_params = list(self.backbone.parameters()) + \
                   list(self.regressor.parameters()) + \
                   list(self.processor.parameters())
        g_opt = torch.optim.AdamW(
            g_params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        
        return [g_opt]
