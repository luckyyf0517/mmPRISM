import torch
import torch.nn as nn

import sys
sys.path.append('.')

from src.model.encoder.pose_encoder import StaticPoseEncoder

class KeypointDiscriminator(nn.Module):
    def __init__(self, num_joints=21, hidden_dim=256, num_heads=8, num_layers=3):
        """
        Keypoint discriminator using Transformer to process hand joints
        Args:
            num_joints (int): Number of hand joints (default: 21)
            hidden_dim (int): Hidden dimension size
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
        """
        super(KeypointDiscriminator, self).__init__()
        
        # Embed 3D coordinates to higher dimension
        self.embed = nn.Linear(3, hidden_dim)
        
        # Position encoding for joints
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_joints, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator
        Args:
            joints (torch.Tensor): Tensor of shape (B, 21, 3) containing hand joint coordinates
        Returns:
            torch.Tensor: Discriminator output with shape (B, 1)
        """
        batch_size = joints.shape[0]
        
        # Embed the 3D coordinates
        x = self.embed(joints)  # (B, 21, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Pass through transformer
        x = self.transformer(x)  # (B, 21, hidden_dim)
        
        # Flatten and classify
        x = x.reshape(batch_size, -1)  # (B, 21 * hidden_dim)
        x = self.classifier(x)  # (B, 1)
        
        return x