"""
Basic 3D convolution block with optional enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct3D(nn.Module):
    """Basic 3D convolution block with GroupNorm and activation.
    
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        stride (int): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input.
            Default: None (kernel_size // 2).
        groups (int): Number of blocked connections from input channels to output channels. Default: 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=None, groups=1):
        super().__init__()
        padding = padding or kernel_size // 2
        
        # Standard convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                            stride, padding, groups=groups, bias=False)
        
        self.norm = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        """Forward function."""
        return self.act(self.norm(self.conv(x))) 