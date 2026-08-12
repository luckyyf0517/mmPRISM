"""
3D Channel attention module with dual-path pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention3D(nn.Module):
    """3D Channel attention module with global pooling and FC layers.
    
    This implementation uses both Global Average Pooling and Global Max Pooling
    for richer feature representation.
    
    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Channel reduction ratio. Default: 16.
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        
        # FC layers for channel attention
        reduced_channels = max(channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward function."""
        B, C = x.shape[:2]
        
        # Global pooling and feature fusion
        avg_pool = self.gap(x).view(B, C)
        max_pool = self.gmp(x).view(B, C)
        pool_concat = torch.cat([avg_pool, max_pool], dim=1)
        
        # Generate attention weights
        weights = self.fc(pool_concat).view(B, C, 1, 1, 1)
        
        return x * weights 