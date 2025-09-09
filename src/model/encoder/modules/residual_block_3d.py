"""
Enhanced 3D Residual Block with multiple attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """Residual block with multiple attention mechanisms.
    
    This block supports multiple attention mechanisms that can be enabled independently:
    - Original Channel + Spatial Attention (use_attention)
    - SE Attention (use_se_attention) - lightweight channel attention
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        use_attention (bool): Whether to use original channel+spatial attention
        use_se_attention (bool): Whether to use SE attention
    """
    def __init__(self, in_channels, out_channels, use_attention=True, 
                 use_se_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.use_se_attention = use_se_attention
        
        # Main conv layers
        from .conv_bn_act_3d import ConvBNAct3D
        self.conv1 = ConvBNAct3D(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels // 4), out_channels)
        )
        
        # Original attention mechanisms (more complex, dual-path)
        if use_attention:
            from .channel_attention_3d import ChannelAttention3D
            from .spatial_attention_3d import SpatialAttention3D
            self.channel_attention = ChannelAttention3D(out_channels)
            self.spatial_attention = SpatialAttention3D()
        
        # SE Attention (lightweight, single-path)
        if use_se_attention:
            from .se_attention_3d import SEAttention3D
            self.se_attention = SEAttention3D(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            shortcut_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut = nn.Sequential(
                shortcut_conv,
                nn.GroupNorm(min(32, out_channels // 4), out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        """Forward function with attention application order.
        
        Attention application order:
        1. Original Channel+Spatial attention (if enabled) - fine-grained attention
        2. SE attention (if enabled) - global channel recalibration
        
        This order allows SE to act as a final recalibration step after detailed attention.
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply original attention mechanisms first (fine-grained)
        if self.use_attention:
            out = self.channel_attention(out)
            out = self.spatial_attention(out)
        
        # Apply SE attention second (global recalibration)
        if self.use_se_attention:
            out = self.se_attention(out)
        
        # Residual connection
        out += identity
        
        return F.silu(out) 