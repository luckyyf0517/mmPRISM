"""
3D Squeeze-and-Excitation attention module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention3D(nn.Module):
    """3D Squeeze-and-Excitation attention module.
    
    SE blocks adaptively recalibrate channel-wise feature responses
    by explicitly modelling interdependencies between channels.
    
    This is a lightweight alternative to ChannelAttention3D that uses
    only Global Average Pooling (more efficient).
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for squeeze operation
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        reduced_channels = max(channels // reduction, 1)
        
        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        
        # Excitation: Two FC layers with ReLU and Sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Forward pass of SE attention."""
        B, C, H, W, D = x.shape
        
        # Squeeze: Global spatial information into channel descriptor
        y = self.squeeze(x).view(B, C)
        
        # Excitation: Channel attention weights
        y = self.excitation(y).view(B, C, 1, 1, 1)
        
        # Scale: Apply attention weights to input features
        return x * y.expand_as(x) 