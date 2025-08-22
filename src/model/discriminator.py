import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from src.model.encoder.pose_encoder import StaticPoseEncoder
from torch.nn.utils import spectral_norm

class KeypointDiscriminator(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=768, num_heads=8, num_layers=3, use_spectral_norm=True, use_minibatch_std=True):
        """
        Keypoint discriminator using StaticPoseEncoder, Spatial Position Encoding and Transformer
        
        Args:
            hidden_dim (int): Hidden dimension size for pose encoder
            output_dim (int): Output dimension from pose encoder
            num_heads (int): Number of attention heads in transformer
            num_layers (int): Number of transformer layers
        """
        super(KeypointDiscriminator, self).__init__()
        
        self.use_minibatch_std = use_minibatch_std
        self.use_spectral_norm = use_spectral_norm
        
        # Static pose encoder to extract joint features
        self.pose_encoder = StaticPoseEncoder(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Spatial position encoding layer: maps 3D coordinates to feature dimension
        self.spatial_encoding = nn.Linear(3, output_dim)
        if self.use_spectral_norm:
            self.spatial_encoding = spectral_norm(self.spatial_encoding)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        # Stack multiple transformer encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Final classification layers with dropout for regularization
        classifier_input_dim = output_dim + (1 if self.use_minibatch_std else 0)
        fc1 = nn.Linear(classifier_input_dim, 1024)
        fc2 = nn.Linear(1024, 256)
        fc3 = nn.Linear(256, 1)
        if self.use_spectral_norm:
            fc1 = spectral_norm(fc1)
            fc2 = spectral_norm(fc2)
            fc3 = spectral_norm(fc3)
        self.classifier = nn.Sequential(
            fc1,
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(0.1),
            fc2,
            nn.LeakyReLU(0.2, inplace=False),
            fc3
        )
        
        # Initialize network weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize network weights using Xavier uniform initialization
        for better training stability
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # If spectral norm is used, m may be wrapped; access weight via .weight
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator
        
        Args:
            joints (torch.Tensor): Hand joint coordinates tensor of shape (B, 21, 3)
                                 where B is batch size, 21 is number of joints,
                                 and 3 is for XYZ coordinates
        
        Returns:
            torch.Tensor: Discriminator scores of shape (B, 1)
        """
        # Get joint features from pose encoder
        joint_features = self.pose_encoder(joints)  # Shape: (B, 21, output_dim)
        
        # Generate spatial position encodings from 3D coordinates
        spatial_pos_encoding = self.spatial_encoding(joints)  # Shape: (B, 21, output_dim)
        
        # Add spatial position encodings to joint features
        # This allows transformer to be aware of the spatial relationships between joints
        enhanced_features = joint_features + spatial_pos_encoding
        
        # Pass through transformer to model joint relationships
        # Transformer will learn the dependencies between different joints
        transformed_features = self.transformer(enhanced_features)  # Shape: (B, 21, output_dim)
        
        # Aggregate features across joints using adaptive max pooling
        # This combines information from all joints into a single feature vector by taking maximum values
        pooled_features = F.adaptive_max_pool1d(transformed_features.transpose(1, 2), 1).squeeze(-1)  # Shape: (B, output_dim)
        
        # Minibatch standard deviation feature (improves discriminator's ability to detect mode collapse)
        if self.use_minibatch_std and pooled_features.size(0) > 1:
            # Compute standard deviation across the batch for each feature, then average to a scalar
            batch_std = torch.sqrt(pooled_features.var(dim=0, unbiased=False) + 1e-8)  # (output_dim,)
            std_mean = batch_std.mean().unsqueeze(0).unsqueeze(0)  # (1, 1)
            std_feat = std_mean.expand(pooled_features.size(0), 1)  # (B, 1)
            pooled_features = torch.cat([pooled_features, std_feat], dim=1)
        elif self.use_minibatch_std:
            # If batch size is 1, append zero std feature
            zeros_feat = torch.zeros(pooled_features.size(0), 1, device=pooled_features.device, dtype=pooled_features.dtype)
            pooled_features = torch.cat([pooled_features, zeros_feat], dim=1)
        
        # Get final discriminator score through classifier (raw score, no sigmoid)
        scores = self.classifier(pooled_features)  # Shape: (B, 1)
        
        return scores
