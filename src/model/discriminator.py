import torch
import torch.nn as nn

import sys
sys.path.append('.')

from src.model.encoder.pose_encoder import StaticPoseEncoder

class KeypointDiscriminator(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=768, num_heads=8, num_layers=3):
        """
        Keypoint discriminator using StaticPoseEncoder and Transformer
        Args:
            hidden_dim (int): Hidden dimension size for pose encoder
            output_dim (int): Output dimension from pose encoder
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
        """
        super(KeypointDiscriminator, self).__init__()
        
        # Use StaticPoseEncoder instead of embed layer
        self.pose_encoder = StaticPoseEncoder(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim*4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
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
            joints (torch.Tensor): Tensor of shape (B, 2, 24, 3) containing hand joint coordinates
        Returns:
            torch.Tensor: Discriminator output with shape (B, 1)
        """
        batch_size = joints.shape[0]
        
        # Get features from pose encoder
        x = self.pose_encoder(joints)  # (B, N, C)
        
        # Pass through transformer (add sequence dimension of length 1)
        x = self.transformer(x)  # (B, N, C)
        
        # Classify
        x = self.classifier(x)  # (B, 1)
        
        return x
