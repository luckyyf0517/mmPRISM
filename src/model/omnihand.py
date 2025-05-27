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
from src.model.discriminator import KeypointDiscriminator


class OmniHand(LightningModule):
    """Base model class with common functionality for all hand reconstruction models."""
    
    def __init__(self, cfg=None):
        super().__init__()
        # Tell lightning to not handle the optimization
        self.automatic_optimization = False
        
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        
        self.simulator = Simulation()
        self.processor = Processor(learnable_weights=cfg.get('learnable_weights', False))
        
        # Initialize backbone (Vision Transformer)
        self.backbone = instantiate_from_config(cfg.backbone)
        if cfg.backbone.pretrained is not None:
            self.backbone.load_state_dict(torch.load(
                os.path.join(cfg.backbone.pretrained, 'model.pth'), weights_only=True), strict=False)
        if cfg.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension from backbone params (support both CubeNet and RTMEncoder3D)
        if hasattr(cfg.backbone.params, 'hidden_dims'):
            self.feature_dim = cfg.backbone.params.hidden_dims[-1]  # For CubeNet
        elif hasattr(cfg.backbone.params, 'stage_channels'):
            self.feature_dim = cfg.backbone.params.stage_channels[-1]  # For RTMEncoder3D
        else:
            raise ValueError("Backbone params must have either 'hidden_dims' or 'stage_channels'")
        
        self.relu = nn.ReLU()
        self.regressor = instantiate_from_config(cfg.regressor)
            
        # Initialize discriminator if enabled
        self.use_discriminator = cfg.training.get('use_discriminator', False)
        self.freeze_discriminator = cfg.discriminator.freeze
        if self.use_discriminator:
            self.discriminator = KeypointDiscriminator(
                hidden_dim=256,
                output_dim=256,
                num_heads=8,
                num_layers=3
            )
            if cfg.discriminator.pretrained is not None:
                self.discriminator.load_state_dict(torch.load(
                    os.path.join(cfg.discriminator.pretrained), weights_only=True), strict=False)
            if self.freeze_discriminator:
                for param in self.discriminator.parameters():
                    param.requires_grad = False
        
        # Training parameters
        self.lr = cfg.training.lr
        self.betas = cfg.training.betas
        self.weight_decay = cfg.training.weight_decay
        self.disc_lr = cfg.training.get('disc_lr', self.lr)
        
        # Loss weights and GAN training schedule
        self.lambda_gan = cfg.training.get('lambda_gan', 0.1)
        # Number of iterations before starting generator GAN training
        self.gan_start_iter = cfg.training.get('gan_start_iter', 0)

    def training_step(self, batch, batch_idx):
        # Forward pass once
        if hasattr(self.regressor, 'encode_keypoints'): 
            batch = self._discretize_gt_poses(batch)
        results = self.forward(batch)
        
        # Calculate reconstruction loss
        rec_loss_dict = self.compute_loss(results, batch)
        # Train generator
        if self.use_discriminator:
            g_opt, d_opt = self.optimizers()
        
            # Calculate GAN losses
            gan_loss_dict = self.compute_disc_loss(results, batch)
            rec_loss_dict.update(gan_loss_dict)
            
            # Add GAN loss to total loss only after warm-up period
            if self.global_step >= self.gan_start_iter:
                rec_loss_dict['loss'] = rec_loss_dict['loss'] + self.lambda_gan * gan_loss_dict['loss_gan_g']
        else:
            g_opt = self.optimizers()

        g_opt.zero_grad()
        self.manual_backward(rec_loss_dict['loss'])
        g_opt.step()
        
        if self.use_discriminator and not self.freeze_discriminator:
            # Train discriminator
            d_opt.zero_grad()
            self.manual_backward(gan_loss_dict['loss_gan_d'])
            d_opt.step()
            
        # Logging
        self._log_info(rec_loss_dict, phase='train')
        self._log_progress(batch_idx, rec_loss_dict)
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        if hasattr(self.regressor, 'encode_keypoints'): 
            batch = self._discretize_gt_poses(batch)
        results = self.forward(batch)  # [B, 2, 24, 3]
        # Compute losses
        loss_dict = self.compute_loss(results, batch)
        # Logging
        self._log_info(loss_dict, phase='valid')
        self._log_progress(batch_idx, loss_dict)
        return loss_dict['loss']  

    def forward(self, input_data):
        """Forward pass
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Dictionary containing either 'joints' or 'simcc_logits'
        """
        features = self.encode_feature(input_data['joints'], input_data['velocities'])
        regressor_output = self.forward_feature(features)
        
        # Check if using RTM decoder (SimCC) or traditional decoder
        if isinstance(regressor_output, tuple) and len(regressor_output) == 3:
            # RTM decoder: returns (simcc_x, simcc_y, simcc_z)
            return {
                'simcc_logits': regressor_output,
            }
        else:
            # Traditional decoder: returns joint coordinates
            return {
                'joints': regressor_output,
            }
        
    def encode_feature(self, joints, velocities):
        """Encode input data into features"""
        x = self.simulator(joints, velocities)
        x = self.processor(x)
        # Extract features using backbone
        features = self.backbone(x)
        return features
    
    def forward_feature(self, features):
        """Forward pass for feature extraction"""
        features = self.relu(features)
        
        # Check if using RTM decoder (SimCC) or traditional decoder
        regressor_output = self.regressor(features)
        
        if isinstance(regressor_output, tuple) and len(regressor_output) == 3:
            # RTM decoder: returns (simcc_x, simcc_y, simcc_z)
            return regressor_output
        else:
            # Traditional decoder: returns joint coordinates
            joints = regressor_output.view(-1, 2, 24, 3)
            return joints
            
    def compute_loss(self, results, batch):
        """Calculate reconstruction losses
        Args:
            results: Model output containing either 'joints' or 'simcc_logits'
            batch: Ground truth data
        Returns:
            Dictionary containing reconstruction losses
        """
        loss_dict = {}
        
        # Check if using RTM decoder (SimCC) or traditional decoder
        if 'simcc_logits' in results:
            # RTM decoder: SimCC loss
            loss_dict = self._compute_simcc_loss(results, batch)
        else:
            # Traditional decoder: coordinate loss
            loss_dict = self._compute_coordinate_loss(results, batch)
        
        return loss_dict
    
    def _discretize_gt_poses(self, batch):
        """Discretize ground truth poses to SimCC grid indices
        Args:
            batch: Dictionary containing 'joints' [B, 2, 24, 3]
        Returns:
            batch: Updated batch with 'joints_grid' containing discretized indices
        """
        target = batch['joints']  # [B, 2, 24, 3]
        
        # Reshape target to match output format [B, 48, 3]
        B = target.shape[0]
        target_reshaped = target.reshape(B, -1, 3)  # [B, 48, 3]
        
        # Use regressor's encode_keypoints method if available (for SimCC decoder)
        target_x_idx, target_y_idx, target_z_idx, valid_mask = self.regressor.encode_keypoints(target_reshaped)
        
        # Stack indices into a single tensor [B, 48, 3]
        joints_indices = torch.stack([target_x_idx, target_y_idx, target_z_idx], dim=-1)
        joints_grid = self.regressor.decode_keypoints_from_indices(joints_indices)
        joints_grid = joints_grid.reshape(-1, 2, 24, 3)
            
        # Add to batch
        batch['joints_grid'] = joints_grid # [B, 2, 24, 3]
        batch['joints_indices'] = joints_indices # [B, 48, 3]   
        batch['joints_valid_mask'] = valid_mask # [B, 48]
        return batch
    
    def _compute_simcc_loss(self, results, batch):
        """Calculate SimCC losses for RTM decoder
        Args:
            results: Contains 'simcc_logits' = (simcc_x, simcc_y, simcc_z)
            batch: Ground truth data with 'joints' [B, 2, 24, 3] and 'joints_grid' [B, 48, 3]
        Returns:
            Dictionary containing SimCC losses
        """
        def compute_dim_loss(pred, target, valid_mask, dim_name):
            """Compute loss for single dimension
            Args:
                pred: Prediction logits [B, 48, simcc_dim]
                target: Target indices [B, 48]
                valid_mask: Valid keypoint mask [B, 48]
                dim_name: Dimension name for logging ('x', 'y', 'z')
            Returns:
                loss: Cross entropy loss for this dimension
            """
            # Flatten for loss calculation
            pred_flat = pred[valid_mask]  # [N_valid, simcc_dim]
            target_flat = target[valid_mask]  # [N_valid]
            
            # Cross-entropy loss
            loss = F.cross_entropy(pred_flat, target_flat)
            return loss

        loss_dict = {}
        simcc_x, simcc_y, simcc_z = results['simcc_logits']  # Each: [B, 48, simcc_dim]
        
        joints_indices = batch['joints_indices']  # [B, 48, 3]
        valid_mask = batch['joints_valid_mask']  # [B, 48]
        
        # Extract target indices
        target_x_idx = joints_indices[..., 0]  # [B, 48]
        target_y_idx = joints_indices[..., 1]  # [B, 48]
        target_z_idx = joints_indices[..., 2]  # [B, 48]
        
        # Calculate cross-entropy losses only for valid keypoints
        if valid_mask.sum() > 0:
            # Compute losses for each dimension
            loss_dict['loss_simcc_x'] = compute_dim_loss(simcc_x, target_x_idx, valid_mask, 'x')
            loss_dict['loss_simcc_y'] = compute_dim_loss(simcc_y, target_y_idx, valid_mask, 'y')
            loss_dict['loss_simcc_z'] = compute_dim_loss(simcc_z, target_z_idx, valid_mask, 'z')
            
            # Total SimCC loss
            loss_dict['loss_simcc'] = (loss_dict['loss_simcc_x'] + 
                                      loss_dict['loss_simcc_y'] + 
                                      loss_dict['loss_simcc_z'])
            
            # For evaluation: decode coordinates and calculate MPJPE
            with torch.no_grad():
                pred_coords, _ = self.regressor.decode_keypoints(simcc_x, simcc_y, simcc_z)  # [B, 48, 3]
                target_coords = batch['joints_grid'].reshape(batch['joints_grid'].shape[0], -1, 3)  # [B, 48, 3]
                
                # Convert to millimeters for MPJPE calculation
                pred_coords = pred_coords * 1e3
                target_coords = target_coords * 1e3
                
                # Calculate MPJPE for valid keypoints only
                pred_valid = pred_coords[valid_mask]
                target_valid = target_coords[valid_mask]
                loss_dict['MPJPE'] = torch.norm(pred_valid - target_valid, dim=-1).mean()
        else:
            raise ValueError("No valid keypoints, please check the input data")

        loss_dict['loss'] = loss_dict['loss_simcc']
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
        loss_dict['MPJPE'] = torch.norm(pred_valid - target_valid, dim=-1).mean()
        
        loss_dict['loss'] = loss_dict['loss_joints']
        return loss_dict
    
    def compute_disc_loss(self, results, batch):
        """Calculate discriminator and adversarial losses
        Args:
            pred: Predicted joint positions [B, 2, 24, 3]
            target: Ground truth joint positions [B, 2, 24, 3]
        Returns:
            Dictionary containing GAN losses
        """
        loss_dict = {}
        pred = results['joints']  # [B, 2, 24, 3]
        target = batch['joints']  # [B, 2, 24, 3]
        
        # Extract last 21 joints for discriminator
        pred_disc = pred[..., 3:, :].reshape(-1, 21, 3)  # [B*2, 21, 3]
        target_disc = target[..., 3:, :].reshape(-1, 21, 3)

        # Discriminator outputs
        d_pred_detach = self.discriminator(pred_disc.detach())
        d_pred = self.discriminator(pred_disc)
        d_target = self.discriminator(target_disc)
        
        # Add margin to help discriminator training
        margin = 0.5
        target_real = 1.0 + margin  # Target for real samples
        target_fake = 0.0 - margin  # Target for fake samples
        
        # Generator loss - do not detach
        loss_dict['loss_gan_g'] = ((d_pred - target_real) ** 2).mean()
        
        # Discriminator loss - detached
        d_loss_real = ((d_target - target_real) ** 2).mean()  # Make discriminator output 1+margin for real samples
        d_loss_fake = ((d_pred_detach - target_fake) ** 2).mean()  # Make discriminator output 0-margin for fake samples
        loss_dict['loss_gan_d'] = d_loss_real + d_loss_fake
        loss_dict['loss_gan_d_real'] = d_loss_real
        loss_dict['loss_gan_d_fake'] = d_loss_fake
        
        # Add discriminator accuracy for monitoring
        with torch.no_grad():
            d_real_acc = (d_target > 0.5).float().mean()
            d_fake_acc = (d_pred <= 0.5).float().mean()
            loss_dict['acc_real'] = d_real_acc
            loss_dict['acc_fake'] = d_fake_acc
        
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
        """Configure optimizers for generator and discriminator"""
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
        
        if self.use_discriminator:
            # Discriminator optimizer
            d_opt = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.disc_lr,
                betas=self.betas,
                weight_decay=0.0
            )
            return [g_opt, d_opt]
        else:
            return [g_opt]
