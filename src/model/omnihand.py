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
        
        # Initialize regression heads for hand keypoints
        self.feature_dim = cfg.backbone.params.hidden_dims[-1]
        self.relu = nn.ReLU()
        self.regressor = nn.ModuleDict({
            'l': nn.Linear(self.feature_dim, 24 * 3),
            'r': nn.Linear(self.feature_dim, 24 * 3)})
            
        # Initialize discriminator if enabled
        self.use_discriminator = cfg.training.get('use_discriminator', False)
        if self.use_discriminator:
            self.discriminator = KeypointDiscriminator(
                hidden_dim=256,
                num_heads=8,
                num_layers=3
            )
            if cfg.discriminator.pretrained is not None:
                self.discriminator.load_state_dict(torch.load(
                    os.path.join(cfg.discriminator.pretrained, 'model.pth'), weights_only=True), strict=False)
            if cfg.discriminator.freeze:
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
        self.gan_start_iter = cfg.training.get('gan_start_iter', 1000)

    def load_state_dict(self, state_dict, strict=False):
            """Override load_state_dict to handle weight migration
            
            Args:
                state_dict: State dict containing model weights
                strict: Whether to strictly enforce that the keys match
            """
            if 'simulator.bm_weights' in state_dict:
                processor_weights = state_dict.pop('simulator.bm_weights')
                state_dict['processor.bm_weights'] = processor_weights
                print("Migrating beamforming weights from simulator to processor")
                
            return super().load_state_dict(state_dict, strict=True)

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
        x = self.processor(x)
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
            
    def compute_loss(self, results, batch):
        """Calculate reconstruction losses
        Args:
            pred: Predicted joint positions [B, 2, 24, 3]
            target: Ground truth joint positions [B, 2, 24, 3]
        Returns:
            Dictionary containing reconstruction losses
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

    def training_step(self, batch, batch_idx):
        # Forward pass once
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
        
        if self.use_discriminator:
            # Train discriminator
            d_opt.zero_grad()
            self.manual_backward(gan_loss_dict['loss_gan_d'])
            d_opt.step()
            
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
    