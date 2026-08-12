"""
3D Spatial attention module using channel statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention3D(nn.Module):
    """3D Spatial attention module using channel statistics and convolution.
    
    Args:
        kernel_size (int): Kernel size of the convolution. Default: 7.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W, D]
        Returns:
            torch.Tensor: Spatially attended tensor [B, C, H, W, D]
        """
        # Compute mean and max along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Generate spatial attention map
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        return x * attention 