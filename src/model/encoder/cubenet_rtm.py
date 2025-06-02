import os

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tools import get_obj_from_str
from src.model.encoder.cubenet import CubeNet


def get_norm_layer(norm_layer, num_features):
    if norm_layer == nn.BatchNorm3d:
        return nn.BatchNorm3d(num_features)
    elif norm_layer == nn.GroupNorm:
        # Use 8 groups for GroupNorm, ensure num_features is divisible by 8
        num_groups = min(8, num_features)
        while num_features % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_features)
    else:
        raise ValueError(f"{norm_layer} is not supported")


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable convolution for 3D inputs.
    
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the depthwise conv.
        stride (int): The stride of the depthwise conv. Default: 1.
        padding (int): The padding of the depthwise conv. Default: 0.
        norm_layer (nn.Module): Normalization layer. Default: nn.BatchNorm3d.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        
        self.depthwise = ConvBNAct3D(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
            groups=in_channels
        )
        
        self.pointwise = ConvBNAct3D(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer, 
        )
    
    def forward(self, x):
        """Forward function."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class ChannelAttention3D(nn.Module):
    """Channel attention module for 3D tensors.
    
    Args:
        channels (int): Number of input channels.
        reduction_ratio (int): Channel reduction ratio. Default: 16.
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward function."""
        att = self.avg_pool(x)
        att = self.fc(att)
        return x * att


class ConvBNAct3D(nn.Module):
    """Basic 3D convolution block with BatchNorm and activation.
    
    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        stride (int): Stride of the convolution. Default: 1.
        padding (int, optional): Zero-padding added to both sides of the input.
            Default: None (kernel_size // 2).
        groups (int): Number of blocked connections from input channels to output channels. Default: 1.
        norm_layer (nn.Module): Normalization layer. Default: nn.BatchNorm3d.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=None, groups=1, norm_layer=nn.BatchNorm3d):
        super().__init__()
        padding = padding or kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                             stride, padding, groups=groups, bias=False)
        self.bn = get_norm_layer(norm_layer, out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        """Forward function."""
        return self.act(self.bn(self.conv(x)))


class DarknetBottleneck3D(nn.Module):
    """The basic bottleneck block used in Darknet for 3D inputs.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv3d, BN3d, and SiLU.
    The first convLayer has filter size of 1x1x1 and the second one has the
    filter size of 3x3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expandsion (float): The kernel size of the convolution. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        norm_layer (nn.Module): Normalization layer. Defaults to nn.BatchNorm3d.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expandsion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        hidden_channels = int(out_channels * expandsion)
        conv = DepthwiseSeparableConv3d if use_depthwise else ConvBNAct3D
        
        self.conv1 = ConvBNAct3D(in_channels, hidden_channels, 1,
                                norm_layer=norm_layer)
        self.conv2 = conv(hidden_channels, out_channels, 3,
                         stride=1, padding=1,
                         norm_layer=norm_layer)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPNeXtBlock3D(nn.Module):
    """The basic bottleneck block used in CSPNeXt for 3D inputs.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expandsion (float): Expand ratio of the hidden channel. Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out. Only works
            when in_channels == out_channels. Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.
        norm_layer (nn.Module): Normalization layer. Defaults to nn.BatchNorm3d.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expandsion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        hidden_channels = int(out_channels * expandsion)
        conv = DepthwiseSeparableConv3d if use_depthwise else ConvBNAct3D
        
        self.conv1 = conv(in_channels, hidden_channels, 3,
                         stride=1, padding=1,
                         norm_layer=norm_layer)
        self.conv2 = DepthwiseSeparableConv3d(
            hidden_channels, out_channels, kernel_size,
            stride=1, padding=kernel_size // 2,
            norm_layer=norm_layer)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPBlock3D(nn.Module):
    """Cross Stage Partial Block for 3D inputs.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expandsion (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        use_cspnext_block (bool): Whether to use CSPNeXt block.
            Defaults to False.
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        norm_layer (nn.Module): Normalization layer. Defaults to nn.BatchNorm3d.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expandsion: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = True,
                 channel_attention: bool = False,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        block = CSPNeXtBlock3D if use_cspnext_block else DarknetBottleneck3D
        mid_channels = int(out_channels * expandsion)
        self.channel_attention = channel_attention
        
        # Main conv and short conv
        self.main_conv = ConvBNAct3D(in_channels, mid_channels, 1,
                                    norm_layer=norm_layer)
        self.short_conv = ConvBNAct3D(in_channels, mid_channels, 1,
                                     norm_layer=norm_layer)
        
        # Final conv
        self.final_conv = ConvBNAct3D(2 * mid_channels, out_channels, 1,
                                     norm_layer=norm_layer)

        # Sequential blocks
        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                expandsion=1.0,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                norm_layer=norm_layer
            ) for _ in range(num_blocks)
        ])
        
        # Channel attention module
        if channel_attention:
            self.attention = ChannelAttention3D(2 * mid_channels)

    def forward(self, x):
        """Forward function."""
        # Short path
        x_short = self.short_conv(x)

        # Main path
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        # Feature fusion
        x_final = torch.cat((x_main, x_short), dim=1)

        # Apply channel attention if enabled
        if self.channel_attention:
            x_final = self.attention(x_final)
            
        return self.final_conv(x_final)


class SPP3D(nn.Module):
    """3D Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7), norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.conv1 = ConvBNAct3D(in_channels, in_channels // 2, 1, 1, 0, norm_layer)
        self.pools = nn.ModuleList([
            nn.MaxPool3d(k, stride=1, padding=k//2) for k in kernel_sizes
        ])
        self.conv2 = ConvBNAct3D(in_channels // 2 * (len(kernel_sizes) + 1), out_channels, 1, 1, 0, norm_layer)
    
    def forward(self, x):
        x = self.conv1(x)
        pool_outs = [x] + [pool(x) for pool in self.pools]
        x = torch.cat(pool_outs, dim=1)
        return self.conv2(x)


class CSPPAFPN3D(nn.Module):
    """CSPNeXt Path Aggregation Feature Pyramid Network for 3D"""
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=None,
                 out_indices=(1, 2),
                 num_csp_blocks=2,
                 expandsion=0.5,
                 norm_layer=nn.BatchNorm3d,
                 spp_kernel_sizes=(3, 5, 7)):
        super().__init__()
        
        self.out_indices = out_indices
        out_channels = out_channels or in_channels[0]
        self.in_channels = in_channels
        
        # SPP block on the last input
        self.spp = SPP3D(in_channels[-1], in_channels[-1], norm_layer=norm_layer, kernel_sizes=spp_kernel_sizes)
        
        # Top-down path
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvBNAct3D(in_channels[idx], in_channels[idx-1], 1, norm_layer=norm_layer))
            self.top_down_blocks.append(
                CSPBlock3D(in_channels[idx-1] * 2, in_channels[idx-1],
                          num_blocks=num_csp_blocks,
                          expandsion=expandsion,
                          norm_layer=norm_layer))
        
        # Bottom-up path
        self.downsample_layers = nn.ModuleList()
        self.trans_layers = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        
        for idx in range(len(in_channels) - 1):
            current_channels = in_channels[idx]
            next_channels = in_channels[idx + 1]
            concat_channels = current_channels + next_channels
            
            self.downsample_layers.append(
                ConvBNAct3D(current_channels, current_channels, 
                           3, stride=2, norm_layer=norm_layer))
            
            self.trans_layers.append(
                ConvBNAct3D(concat_channels, next_channels, 
                           1, stride=1, norm_layer=norm_layer)
            )
            
            self.bottom_up_blocks.append(
                CSPBlock3D(next_channels, next_channels,
                          num_blocks=num_csp_blocks,
                          expandsion=expandsion,
                          norm_layer=norm_layer))
    
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Multi-level features from the backbone
        
        Returns:
            tuple[Tensor]: Multi-level features after FPN
        """
        # Apply SPP on the last input
        inputs[-1] = self.spp(inputs[-1])
        
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


class CSPEncoder3D(nn.Module):
    """CSPEncoder3D with CSPNeXt architecture for 3D inputs"""
    def __init__(self, 
                 in_channels=32,
                 base_channels=64,
                 stage_channels=[128, 256, 512, 1024], 
                 stage_blocks=[2, 4, 4, 2],  # Changed to standard P5 architecture
                 spp_kernel_sizes=(3, 5, 7),
                 expandsion=0.5,
                 channel_attention=True,
                 norm_layer='torch.nn.BatchNorm3d',
                 **kwargs):
        super().__init__()
        
        # Convert norm_layer string to actual class if needed
        norm_layer = get_obj_from_str(norm_layer)
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.stage_channels = stage_channels
        self.stage_blocks = stage_blocks
        
        # Stem layer with focus-like structure
        self.stem = nn.Sequential(
            ConvBNAct3D(in_channels, base_channels // 2, 3, 1, 1, norm_layer=norm_layer),
            ConvBNAct3D(base_channels // 2, base_channels, 3, 2, 1, norm_layer=norm_layer),
            ConvBNAct3D(base_channels, base_channels, 3, 1, 1, norm_layer=norm_layer)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_ch = base_channels
        
        for i, (out_ch, num_blocks) in enumerate(zip(stage_channels, stage_blocks)):
            # Downsample
            downsample = ConvBNAct3D(in_ch, out_ch, 3, 2, 1, norm_layer=norm_layer)
            
            # CSP blocks with RepVGG structure
            csp_block = CSPBlock3D(
                out_ch, out_ch, 
                num_blocks=num_blocks,
                expandsion=expandsion,
                channel_attention=channel_attention and i >= 2,  # Only use attention in later stages
                norm_layer=norm_layer
            )
            
            stage = nn.Sequential(downsample, csp_block)
            self.stages.append(stage)
            in_ch = out_ch
        
        # Add PAFPN
        self.neck = CSPPAFPN3D(
            in_channels=stage_channels[-3:],  # Use features from last 3 stages [256, 512, 1024]
            out_channels=None,
            out_indices=(1, 2),  # Output last two levels
            num_csp_blocks=2,
            expandsion=expandsion,
            norm_layer=norm_layer, 
            spp_kernel_sizes=spp_kernel_sizes
        )
        
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
        
        # Stages
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 1:  # Collect features from stage2 onwards for FPN
                features.append(x)
        
        # Apply PAFPN
        neck_outs = self.neck(features)
        
        # Use the last feature map for global pooling
        x = neck_outs[-1]
        
        # Global feature
        global_feat = self.global_pool(x).flatten(1)
        
        return global_feat
