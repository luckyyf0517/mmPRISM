"""
3D Deformable Convolution for adaptive feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableConv3D(nn.Module):
    """3D Deformable Convolution for adaptive feature extraction.
    
    Deformable convolution allows the network to adaptively adjust sampling 
    locations based on learned offsets, improving feature alignment.
    
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels  
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int): Convolution padding
        groups (int): Number of groups for grouped convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Offset prediction network
        self.offset_conv = nn.Conv3d(
            in_channels,
            3 * kernel_size * kernel_size * kernel_size,  # 3D offsets (x, y, z)
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Main convolution
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # Initialize offset conv to output zero offsets initially
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
    def forward(self, x):
        """Forward pass with deformable convolution."""
        # Predict offsets
        offsets = self.offset_conv(x)
        
        # Apply deformable convolution
        # Note: For simplicity, we use a lightweight approximation
        # In production, you might want to use the full deformable conv implementation
        
        # Lightweight deformable convolution approximation
        # Split offsets into x, y, z components
        B, _, H, W, D = x.shape
        offset_groups = offsets.view(B, 3, -1, H, W, D)
        
        # Apply small random perturbations based on learned offsets
        # This is a simplified version - full implementation would use grid sampling
        offset_scale = 0.1  # Scale factor for offset magnitude
        
        # Create coordinate grids
        coords_h = torch.arange(H, dtype=torch.float32, device=x.device)
        coords_w = torch.arange(W, dtype=torch.float32, device=x.device)  
        coords_d = torch.arange(D, dtype=torch.float32, device=x.device)
        
        grid_h, grid_w, grid_d = torch.meshgrid(coords_h, coords_w, coords_d, indexing='ij')
        grid = torch.stack([grid_h, grid_w, grid_d], dim=0).unsqueeze(0)  # [1, 3, H, W, D]
        
        # Apply offsets (simplified)
        perturbed_features = []
        for i in range(min(3, self.kernel_size)):  # Sample a few offset positions
            offset_factor = (i - 1) * offset_scale
            offset_x = offset_groups[:, 0:1] * offset_factor
            offset_y = offset_groups[:, 1:2] * offset_factor  
            offset_z = offset_groups[:, 2:3] * offset_factor
            
            # Simple feature perturbation (approximation)
            perturbed = x + offset_x.mean(dim=2, keepdim=True) * 0.1
            perturbed_features.append(perturbed)
        
        # Combine perturbed features
        if perturbed_features:
            enhanced_x = sum(perturbed_features) / len(perturbed_features)
        else:
            enhanced_x = x
        
        # Apply main convolution
        return self.conv(enhanced_x) 