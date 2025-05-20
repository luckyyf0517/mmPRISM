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
from src.fmcw.simulator import Simulation


class OmniHand(LightningModule):
    """Base model class with common functionality for all hand reconstruction models."""
    
    def __init__(self, cfg=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        
        self.simulator = Simulation(learnable_weights=cfg.get('learnable_weights', False))
        
        # Initialize backbone (Vision Transformer)
        self.backbone = instantiate_from_config(cfg.backbone)
        if cfg.backbone.pretrained is not None:
            self.backbone.load_state_dict(torch.load(
                os.path.join(cfg.backbone.pretrained, 'model.pth'), weights_only=True), strict=False)
        if cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Initialize regression heads for hand keypoints
        self.feature_dim = cfg.backbone.params.hidden_dims[-1]
        self.relu = nn.ReLU()
        self.regressor = nn.ModuleDict({
            'l': nn.Linear(self.feature_dim, 24 * 3),
            'r': nn.Linear(self.feature_dim, 24 * 3)})
        
        # Training parameters
        self.lr = cfg.training.lr
        self.betas = cfg.training.betas
        self.weight_decay = cfg.training.weight_decay

    def forward(self, input_data):
        """Forward pass
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            joints: Predicted joint positions of shape [B, 2, 24, 3] (2 for l/r hands)
        """
        features = self.encode_feature(input_data['joints'], input_data['velocities'])
        joints = self.forward_feature(features)
        return {
            'joints': joints,
        }
        
    def encode_feature(self, joints, velocities):
        """Encode input data into features"""
        x = self.simulator(joints, velocities)
        # Extract features using ViT
        features = self.backbone(x)  # [B, feature_dim, 4, 4, 4]
        # Global max pooling across spatial dimensions
        features = F.adaptive_max_pool3d(features, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, feature_dim]
        return features
    
    def forward_feature(self, features):
        """Forward pass for feature extraction"""
        features = self.relu(features)
        # Predict joint positions for both hands
        joints_l = self.regressor['l'](features).view(-1, 24, 3)
        joints_r = self.regressor['r'](features).view(-1, 24, 3)
        # Stack l and r hand joints
        joints = torch.stack([joints_l, joints_r], dim=1)  # [B, 2, 24, 3]
        return joints

    def shared_step(self, batch, batch_idx, phase='train'):
        """Shared training/validation/test step"""
        # Forward pass
        results = self.forward(batch)  # [B, 2, 24, 3]
        # Compute losses
        loss_dict = self.compute_loss(results, batch)
        # Logging
        self._log_info(loss_dict, phase)
        # self._log_progress(batch_idx, loss_dict)
        return loss_dict['loss']  
            
    def compute_loss(self, results, batch):
        """Calculate training losses
        Args:
            pred: Predicted joint positions [B, 2, 24, 3]
            target: Ground truth joint positions [B, 2, 24, 3]
        Returns:
            Dictionary containing losses
        """
        loss_dict = {}
        pred = results['joints'] * 1e3
        target = batch['joints'] * 1e3

        # Check if any value in the last dimension (xyz) is NaN
        valid_mask = ~torch.any(torch.isnan(target), dim=-1)

        # Apply mask to both predictions and targets
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # L1 loss on valid joint positions
        loss_dict['loss_joints'] = F.l1_loss(pred_valid, target_valid)
        
        # MPJPE (Mean Per Joint Position Error) on valid joints
        loss_dict['MPJPE'] = torch.norm(pred_valid - target_valid, dim=-1).mean()
        
        # Total loss
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
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, phase='train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, phase='valid')
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, phase='test')
