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


class RepVGGBlock3D(nn.Module):
    """RepVGG-style 3D convolution block with identity mapping"""
    def __init__(self, channels, norm_layer=nn.BatchNorm3d):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 1)
        self.bn1 = get_norm_layer(norm_layer, channels)
        self.bn2 = get_norm_layer(norm_layer, channels)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)) + self.bn2(self.conv2(x)))


class ChannelAttention3D(nn.Module):
    """Channel attention module for 3D tensors"""
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
        att = self.avg_pool(x)
        att = self.fc(att)
        return x * att


class ConvBNAct3D(nn.Module):
    """Basic 3D convolution block with BatchNorm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=None, norm_layer=nn.BatchNorm3d):
        super().__init__()
        padding = padding or kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, 
                             stride, padding, bias=False)
        self.bn = get_norm_layer(norm_layer, out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock3D(nn.Module):
    """Cross Stage Partial Block for 3D tensors with RepVGG-style blocks"""
    def __init__(self, in_channels, out_channels, num_blocks, 
                 expansion=0.5, add_identity=True, 
                 use_attention=True,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = ConvBNAct3D(in_channels, hidden_channels, 1, 1, 0, norm_layer)
        self.conv2 = ConvBNAct3D(in_channels, hidden_channels, 1, 1, 0, norm_layer)
        
        self.blocks = nn.Sequential(*[
            RepVGGBlock3D(hidden_channels, norm_layer)
            for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvBNAct3D(hidden_channels * 2, out_channels, 1, 
                                norm_layer=norm_layer)
        self.attention = ChannelAttention3D(out_channels) if use_attention else None
        self.add_identity = add_identity and in_channels == out_channels
        
    def forward(self, x):
        identity = x if self.add_identity else None
        
        x1 = self.conv1(x)
        x2 = self.blocks(self.conv2(x))
        
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3(x)
        
        if self.attention is not None:
            x = self.attention(x)
            
        if identity is not None:
            x = x + identity
            
        return x


class SPP3D(nn.Module):
    """3D Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), norm_layer=nn.BatchNorm3d):
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
                 expand_ratio=0.5,
                 norm_layer=nn.BatchNorm3d):
        super().__init__()
        
        self.out_indices = out_indices
        out_channels = out_channels or in_channels[0]
        self.in_channels = in_channels
        
        # SPP block on the last input
        self.spp = SPP3D(in_channels[-1], in_channels[-1], norm_layer=norm_layer)
        
        # Top-down path
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvBNAct3D(in_channels[idx], in_channels[idx-1], 1, norm_layer=norm_layer))
            self.top_down_blocks.append(
                CSPBlock3D(in_channels[idx-1] * 2, in_channels[idx-1],
                          num_csp_blocks, expansion=expand_ratio,
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
                          num_csp_blocks, expansion=expand_ratio,
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


class RTMEncoder3D(nn.Module):
    """RTMEncoder3D with CSPNeXt architecture for 3D inputs"""
    def __init__(self, 
                 in_channels=32,
                 base_channels=64,
                 stage_channels=[128, 256, 512, 1024],
                 stage_blocks=[2, 4, 4, 2],  # Changed to standard P5 architecture
                 expansion=0.5,
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
            ConvBNAct3D(in_channels, base_channels // 2, 3, 1, 1, norm_layer),
            ConvBNAct3D(base_channels // 2, base_channels, 3, 2, 1, norm_layer),
            ConvBNAct3D(base_channels, base_channels, 3, 1, 1, norm_layer)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_ch = base_channels
        
        for i, (out_ch, num_blocks) in enumerate(zip(stage_channels, stage_blocks)):
            # Downsample
            downsample = ConvBNAct3D(in_ch, out_ch, 3, 2, 1, norm_layer)
            
            # CSP blocks with RepVGG structure
            csp_block = CSPBlock3D(
                out_ch, out_ch, num_blocks, 
                expansion=expansion,
                use_attention=channel_attention and i >= 2,  # Only use attention in later stages
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
            expand_ratio=expansion,
            norm_layer=norm_layer
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


class PoseDecoderSimCC3D(nn.Module):
    def __init__(self, 
                 input_feature_dim, 
                 output_kpts=42,
                 input_spatial_dims=(32, 32, 32), 
                 simcc_split_ratio=[2.0, 2.0, 2.0],
                 shared_mlp_hidden_dim=512,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__()
        self.output_kpts = output_kpts
        self.input_spatial_dims = input_spatial_dims
        self.simcc_split_ratio = simcc_split_ratio

        # Calculate SimCC dimensions
        self.simcc_x = int(input_spatial_dims[2] * simcc_split_ratio[2])  # Width
        self.simcc_y = int(input_spatial_dims[1] * simcc_split_ratio[1])  # Height  
        self.simcc_z = int(input_spatial_dims[0] * simcc_split_ratio[0])  # Depth

        # Calculate resolution: total range 1.0 (-0.5~0.5) divided by number of grids
        self.res_x = 1.0 / self.simcc_x  # Size of each grid cell
        self.res_y = 1.0 / self.simcc_y
        self.res_z = 1.0 / self.simcc_z

        # Shared MLP for feature processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_feature_dim, shared_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_mlp_hidden_dim, shared_mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # SimCC heads for X, Y, Z coordinates
        self.head_x = nn.Linear(shared_mlp_hidden_dim, output_kpts * self.simcc_x)
        self.head_y = nn.Linear(shared_mlp_hidden_dim, output_kpts * self.simcc_y)
        self.head_z = nn.Linear(shared_mlp_hidden_dim, output_kpts * self.simcc_z)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: Global feature vector [B, feature_dim]
        
        Returns:
            simcc_x: [B, output_kpts, simcc_x] - X coordinate logits
            simcc_y: [B, output_kpts, simcc_y] - Y coordinate logits
            simcc_z: [B, output_kpts, simcc_z] - Z coordinate logits
        """
        # Shared feature processing
        shared_feat = self.shared_mlp(features)
        
        # SimCC predictions
        pred_x = self.head_x(shared_feat)  # [B, output_kpts * simcc_x]
        pred_y = self.head_y(shared_feat)  # [B, output_kpts * simcc_y]
        pred_z = self.head_z(shared_feat)  # [B, output_kpts * simcc_z]
        
        # Reshape to [B, output_kpts, simcc_dim]
        batch_size = pred_x.size(0)
        simcc_x = pred_x.view(batch_size, self.output_kpts, self.simcc_x)
        simcc_y = pred_y.view(batch_size, self.output_kpts, self.simcc_y)
        simcc_z = pred_z.view(batch_size, self.output_kpts, self.simcc_z)
        
        return simcc_x, simcc_y, simcc_z

    def decode_keypoints(self, simcc_x, simcc_y, simcc_z):
        """Decode using resolution for normalized coordinate space [-0.5, 0.5]"""
        batch_size = simcc_x.size(0)
        
        # Get probability distributions
        prob_x = F.softmax(simcc_x, dim=-1)  # [B, output_kpts, simcc_x]
        prob_y = F.softmax(simcc_y, dim=-1)  # [B, output_kpts, simcc_y]
        prob_z = F.softmax(simcc_z, dim=-1)  # [B, output_kpts, simcc_z]
        
        # Create grid coordinates (start from -0.5, increment by resolution)
        device = simcc_x.device
        x_grid = torch.arange(self.simcc_x, dtype=torch.float32, device=device) * self.res_x - 0.5
        y_grid = torch.arange(self.simcc_y, dtype=torch.float32, device=device) * self.res_y - 0.5
        z_grid = torch.arange(self.simcc_z, dtype=torch.float32, device=device) * self.res_z - 0.5
        
        # Calculate expected coordinates using soft-argmax
        x_coords = torch.sum(prob_x * x_grid.view(1, 1, -1), dim=-1)  # [B, output_kpts]
        y_coords = torch.sum(prob_y * y_grid.view(1, 1, -1), dim=-1)  # [B, output_kpts]
        z_coords = torch.sum(prob_z * z_grid.view(1, 1, -1), dim=-1)  # [B, output_kpts]
        
        # Stack coordinates
        keypoints = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # [B, output_kpts, 3]
        
        # Calculate confidence scores (average of max probabilities)
        scores_x = torch.max(prob_x, dim=-1)[0]
        scores_y = torch.max(prob_y, dim=-1)[0]
        scores_z = torch.max(prob_z, dim=-1)[0]
        scores = (scores_x + scores_y + scores_z) / 3.0
        
        return keypoints, scores
    
    def decode_keypoints_from_indices(self, indices):
        """Decode using resolution for normalized coordinate space [-0.5, 0.5]"""
        x_coords = indices[..., 0] * self.res_x - 0.5
        y_coords = indices[..., 1] * self.res_y - 0.5
        z_coords = indices[..., 2] * self.res_z - 0.5
        keypoints = torch.stack([x_coords, y_coords, z_coords], dim=-1)  # [B, output_kpts, 3]
        return keypoints
        

    def encode_keypoints(self, keypoints, valid_mask=None):
        """Encode using resolution for normalized coordinate space [-0.5, 0.5]"""
        # Extract coordinates
        x_coords = keypoints[..., 0]  # [B, output_kpts]
        y_coords = keypoints[..., 1]  # [B, output_kpts]
        z_coords = keypoints[..., 2]  # [B, output_kpts]
        
        # Clamp coordinates to normalized range [-0.5, 0.5]
        x_coords = torch.clamp(x_coords, -0.5, 0.5 - 1e-6)
        y_coords = torch.clamp(y_coords, -0.5, 0.5 - 1e-6)
        z_coords = torch.clamp(z_coords, -0.5, 0.5 - 1e-6)
        
        # Convert coordinates to grid indices: shift to [0,1] range then divide by resolution
        target_x_idx = ((x_coords + 0.5) / self.res_x).long()  # [B, output_kpts]
        target_y_idx = ((y_coords + 0.5) / self.res_y).long()  # [B, output_kpts]
        target_z_idx = ((z_coords + 0.5) / self.res_z).long()  # [B, output_kpts]
        
        # Clamp indices to valid range
        target_x_idx = torch.clamp(target_x_idx, 0, self.simcc_x - 1)
        target_y_idx = torch.clamp(target_y_idx, 0, self.simcc_y - 1)
        target_z_idx = torch.clamp(target_z_idx, 0, self.simcc_z - 1)
        
        # Create or update validity mask
        if valid_mask is None:
            # Create mask for valid keypoints (non-NaN)
            valid_mask = ~torch.any(torch.isnan(keypoints), dim=-1)  # [B, output_kpts]
        
        # Set indices of invalid keypoints to -1
        target_x_idx[~valid_mask] = -1
        target_y_idx[~valid_mask] = -1
        target_z_idx[~valid_mask] = -1

        return target_x_idx, target_y_idx, target_z_idx, valid_mask


if __name__ == "__main__":
    # Test RTMEncoder3D with GroupNorm
    model = RTMEncoder3D(
        in_channels=32,
        base_channels=64,
        stage_channels=[128, 256, 512, 1024],
        stage_blocks=[2, 4, 4, 2],  # Standard P5 architecture
        expansion=0.5,
        channel_attention=True,
        norm_layer='torch.nn.GroupNorm'
    )
    
    # Test input shape: [batch_size, channels, height, width, depth]
    batch_size = 2
    input_tensor = torch.randn(batch_size, 32, 32, 32, 32)
    
    # Forward pass through encoder
    global_feat = model(input_tensor)
    print("=== RTMEncoder3D Test ===")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Global feature shape: {global_feat.shape}")
    
    # Test PoseDecoderSimCC3D
    decoder = PoseDecoderSimCC3D(
        input_feature_dim=1024,  # Should match encoder's output dimension
        output_kpts=48,  # 24 keypoints per hand × 2 hands
        input_spatial_dims=(32, 32, 32),
        simcc_split_ratio=[2.0, 2.0, 2.0],
        shared_mlp_hidden_dim=512
    )
    
    # Forward pass through decoder
    simcc_x, simcc_y, simcc_z = decoder(global_feat)
    print("\n=== PoseDecoderSimCC3D Test ===")
    print(f"SimCC output shapes:")
    print(f"X: {simcc_x.shape}")
    print(f"Y: {simcc_y.shape}")
    print(f"Z: {simcc_z.shape}")
    
    # Test keypoint decoding
    keypoints, scores = decoder.decode_keypoints(simcc_x, simcc_y, simcc_z)
    print("\n=== Keypoint Decoding Test ===")
    print(f"Decoded keypoints shape: {keypoints.shape}")
    print(f"Confidence scores shape: {scores.shape}")
    
    # Test keypoint encoding
    target_keypoints = torch.rand(batch_size, 48, 3) - 0.5  # Random keypoints in [-0.5, 0.5]
    target_x_idx, target_y_idx, target_z_idx, valid_mask = decoder.encode_keypoints(target_keypoints)
    print("\n=== Keypoint Encoding Test ===")
    print(f"Encoded indices shapes:")
    print(f"X indices: {target_x_idx.shape}")
    print(f"Y indices: {target_y_idx.shape}")
    print(f"Z indices: {target_z_idx.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")
    
    print("\n=== Test Completed Successfully! ===")
    print("RTMEncoder3D with complete CSPNeXt + PAFPN architecture is working correctly.")


