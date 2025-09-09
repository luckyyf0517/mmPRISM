import os
import math

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tools import get_obj_from_str
from .modules import (
    ConvBNAct3D, 
    ResidualBlock3D, 
    ChannelAttention3D, 
    SpatialAttention3D,
    SEAttention3D
)


class PAFPN(nn.Module):
    """Path Aggregation Feature Pyramid Network for 3D.
    
    Args:
        in_channels (list): Input channels for each level
        out_channels (int): Output channels
        out_indices (tuple): Output level indices
        num_blocks (int): Number of residual blocks
        use_se_attention (bool): Whether to use SE attention
    """
    def __init__(self,
                 in_channels=[128, 256, 512],
                 out_channels=None,
                 out_indices=(1, 2),
                 num_blocks=2,
                 use_se_attention=False):
        super().__init__()
        
        self.out_indices = out_indices
        out_channels = out_channels or in_channels[0]
        self.in_channels = in_channels
        
        # Top-down path
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvBNAct3D(in_channels[idx], in_channels[idx-1], 1))
            
            # Use enhanced residual blocks
            blocks = []
            for _ in range(num_blocks):
                blocks.append(ResidualBlock3D(
                    in_channels[idx-1] * 2 if len(blocks) == 0 else in_channels[idx-1], 
                    in_channels[idx-1], 
                    use_attention=False,
                    use_se_attention=use_se_attention,
                ))
            self.top_down_blocks.append(nn.Sequential(*blocks))
        
        # Bottom-up path
        self.downsample_layers = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        
        for idx in range(len(in_channels) - 1):
            current_channels = in_channels[idx]
            next_channels = in_channels[idx + 1]
            concat_channels = current_channels + next_channels
            
            self.downsample_layers.append(
                ConvBNAct3D(current_channels, current_channels, 3, stride=2))
            
            self.trans_layers.append(
                ConvBNAct3D(concat_channels, next_channels, 1, stride=1))
            
            # Use enhanced residual blocks
            blocks = []
            for _ in range(num_blocks):
                blocks.append(ResidualBlock3D(
                    next_channels, next_channels, 
                    use_attention=False,
                    use_se_attention=use_se_attention,
                ))
            self.bottom_up_blocks.append(nn.Sequential(*blocks))
    
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Multi-level features from the backbone
        
        Returns:
            tuple[Tensor]: Multi-level features after FPN
        """
        # Top-down path (from high-level to low-level)
        feat = inputs[-1]
        top_down_feats = [feat]
        
        for idx in range(len(inputs) - 1):
            feat = self.reduce_layers[idx](feat)
            feat = self.upsample(feat)
            feat = torch.cat([feat, inputs[-(idx+2)]], dim=1)
            feat = self.top_down_blocks[idx](feat)
            top_down_feats.append(feat)
        
        # Bottom-up path (from low-level to high-level)
        outputs = [top_down_feats[-1]]
        feat = top_down_feats[-1]
        
        for idx in range(len(inputs) - 1):
            feat_down = self.downsample_layers[idx](feat)
            feat_td = top_down_feats[-(idx+2)]
            feat = torch.cat([feat_down, feat_td], dim=1)
            feat = self.trans_layers[idx](feat)
            feat = self.bottom_up_blocks[idx](feat)
            outputs.append(feat)
        
        return tuple(outputs[i] for i in self.out_indices)


class CubeNet(nn.Module):
    """3D Encoder with attention mechanisms and PAFPN neck.
    
    Args:
        in_channels (int): Input channels
        base_channels (int): Base number of channels
        stage_channels (list): Channels for each stage
        stage_blocks (list): Number of blocks for each stage
        use_attention (bool): Whether to use attention mechanisms
        use_pafpn (bool): Whether to use PAFPN neck
        use_se_attention (bool): Whether to use SE attention
    """
    def __init__(self, 
                 in_channels=32,
                 base_channels=64,
                 stage_channels=[64, 128, 256, 512], 
                 stage_blocks=[2, 2, 2, 2],
                 use_attention=True,
                 use_pafpn=True,
                 use_se_attention=False,
                 **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_attention = use_attention
        self.use_pafpn = use_pafpn
        self.use_se_attention = use_se_attention
        
        # Stem layer
        stem_conv = nn.Conv3d(in_channels, base_channels, 
                            kernel_size=3, stride=2, padding=1, bias=False)
        
        self.stem = nn.Sequential(
            stem_conv,
            nn.GroupNorm(min(32, base_channels // 4), base_channels),
            nn.SiLU(inplace=True)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_ch = base_channels
        
        for i, (out_ch, num_blocks) in enumerate(zip(stage_channels, stage_blocks)):
            stage = []
            
            # Downsampling (except for first stage if channels match)
            if i > 0 or in_ch != out_ch:
                downsample_conv = nn.Conv3d(in_ch, out_ch, 
                                          kernel_size=3, 
                                          stride=2 if i > 0 else 1, 
                                          padding=1, bias=False)
                
                stage.append(nn.Sequential(
                    downsample_conv,
                    nn.GroupNorm(min(32, out_ch // 4), out_ch),
                    nn.SiLU(inplace=True)
                ))
                stage_in_ch = out_ch
            else:
                stage_in_ch = in_ch
            
            # Add residual blocks with enhanced modules
            for j in range(num_blocks):
                block_in_ch = stage_in_ch if j == 0 else out_ch
                stage.append(ResidualBlock3D(
                    block_in_ch, out_ch, 
                    use_attention=use_attention,
                    use_se_attention=use_se_attention,
                ))
            
            self.stages.append(nn.Sequential(*stage))
            in_ch = out_ch
        
        # PAFPN neck with enhanced modules
        if self.use_pafpn:
            self.neck = PAFPN(
                in_channels=stage_channels,
                out_channels=None,
                out_indices=(0, 1, 2, 3),  # Output all levels to access the highest level
                num_blocks=2,
                use_se_attention=use_se_attention
            )
        else:
            self.neck = None
        
        # Global average pooling for final feature
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Feature dimension output
        self.feature_dim = stage_channels[-1]
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the network
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W, D]
        
        Returns:
            global_feat (torch.Tensor): Global feature vector [B, feature_dim]
        """
        # Stem
        x = self.stem(x)
        
        # Stages - collect features for PAFPN
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
        
        # PAFPN neck
        if self.use_pafpn:
            neck_outs = self.neck(features)
            # Use the last feature map for global pooling
            x = x + neck_outs[-1]
        else:
            # Use final stage output directly
            x = features[-1]
        
        # Global feature
        global_feat = self.global_pool(x).flatten(1)
        
        return global_feat

