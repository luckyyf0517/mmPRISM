import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tools import get_obj_from_str


class ChannelAttention(nn.Module):
    """Basic channel attention mechanism as described in paper.
    
    Args:
        channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for FC layers
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        
        # Simple FC layer as in paper
        reduced_channels = max(channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W, D]
        Returns:
            torch.Tensor: Channel attended tensor [B, C, H, W, D]
        """
        B, C = x.shape[:2]
        
        # Global pooling and feature fusion
        avg_pool = self.gap(x).view(B, C)
        max_pool = self.gmp(x).view(B, C)
        pool_concat = torch.cat([avg_pool, max_pool], dim=1)
        
        # Generate attention weights
        weights = self.fc(pool_concat).view(B, C, 1, 1, 1)
        
        return x * weights


class SpatialAttention3D(nn.Module):
    """Basic 3D spatial attention as described in paper."""
    def __init__(self, kernel_size=7):
        super().__init__()
        # Simple Conv2 as mentioned in paper
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W, D]
        Returns:
            torch.Tensor: Spatially attended tensor [B, C, H, W, D]
        """
        # Compute mean and max along channel dimension
        mean_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Generate spatial attention map as in paper
        attention = self.sigmoid(self.conv(torch.cat([mean_out, max_out], dim=1)))
        
        return x * attention


class AttentionBlock(nn.Module):
    """Attention residual block with two branches as described in paper.
    
    Branch 1: 1x1 conv to preserve current level features
    Branch 2: Downsampling -> processing -> Upsampling (hourglass)
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        reduction_ratio (int): Channel reduction ratio
    """
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super().__init__()
        
        # Branch 1: 1x1 conv to preserve features of current level
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, out_channels // 4), out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Hourglass structure (downsample -> process -> upsample)
        mid_channels = max(out_channels // 2, 16)
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, mid_channels // 4), mid_channels),
            nn.ReLU(inplace=True)
        )
        self.process = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, mid_channels // 4), mid_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(mid_channels, out_channels, kernel_size=3, stride=2, 
                              padding=1, output_padding=1, bias=False),
            nn.GroupNorm(min(32, out_channels // 4), out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention3D()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(32, out_channels // 4), out_channels)
            )
    
    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W, D]
        Returns:
            torch.Tensor: Output tensor [B, out_channels, H, W, D]
        """
        identity = self.shortcut(x)
        
        # Two branches as described in paper
        branch1_out = self.branch1(x)  # Preserve current level features
        
        # Branch 2: Hourglass structure
        branch2_out = self.downsample(x)
        branch2_out = self.process(branch2_out)
        branch2_out = self.upsample(branch2_out)
        
        # Combine branches
        out = branch1_out + branch2_out
        
        # Apply attention mechanisms
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        # Add identity and activation
        out += identity
        
        return F.relu(out)


class MMHandEncoder(nn.Module):
    """Efficient MMHand Encoder with simplified architecture.
    
    Args:
        input_dim (int): Input channels
        hidden_dims (list): Hidden dimensions for each stage
        num_blocks (list): Number of blocks per stage
        reduction_ratio (int): Channel reduction ratio
    """
    def __init__(self, 
                 input_dim=128,
                 hidden_dims=None,
                 num_blocks=None,
                 reduction_ratio=16,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Initial convolution with stride 2
        self.conv_input = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden_dims[0] // 4), hidden_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_channels = hidden_dims[0]
        
        for stage_idx, (out_channels, num_block) in enumerate(zip(hidden_dims, num_blocks)):
            stage = []
            
            # Downsampling at the beginning of each stage (except first)
            if stage_idx > 0:
                stage.append(nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                             stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(32, out_channels // 4), out_channels),
                    nn.ReLU(inplace=True)
                ))
            
            # Add attention blocks
            for block_idx in range(num_block):
                if block_idx == 0 and stage_idx == 0:
                    stage.append(AttentionBlock(in_channels, out_channels, reduction_ratio))
                else:
                    stage.append(AttentionBlock(out_channels, out_channels, reduction_ratio))
            
            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Feature dimension
        self.feature_dim = hidden_dims[-1]
    
    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input tensor [B, input_dim, H, W, D]
        Returns:
            torch.Tensor: Global feature vector [B, feature_dim]
        """
        # Initial convolution
        x = self.conv_input(x)
        
        # Process through stages
        for stage in self.stages:
            x = stage(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.flatten(1)  # [B, feature_dim]
        
        return x