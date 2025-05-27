import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.tools import get_obj_from_str


def get_norm_layer(norm_layer, num_features):
    if norm_layer == nn.BatchNorm3d:
        return nn.BatchNorm3d(num_features)
    elif norm_layer == nn.GroupNorm:
        return nn.GroupNorm(8, num_features)
    else:
        raise ValueError(f"{norm_layer} is not supported")


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm3d):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = get_norm_layer(norm_layer, planes)
        self.bn2 = get_norm_layer(norm_layer, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            bn = get_norm_layer(norm_layer, self.expansion * planes)
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), bn)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class CubeNet(nn.Module):
    def __init__(self, 
                 block='src.model.encoder.cubenet.BasicBlock3D', 
                 input_dim=128, 
                 hidden_dims=None, 
                 num_blocks=None, 
                 strides=None, 
                 norm_layer=nn.BatchNorm3d, 
                 last_norm=True, **kwargs):
        super(CubeNet, self).__init__()
        
        block = get_obj_from_str(block)
        norm_layer = get_obj_from_str(norm_layer)
        
        self.conv0 = nn.Conv3d(input_dim, hidden_dims[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = get_norm_layer(norm_layer, hidden_dims[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.in_planes = hidden_dims[0]
        self.layers = nn.Sequential(
            *[self._make_layer(block, hidden_dims[i], num_blocks[i], stride=strides[i], norm_layer=norm_layer) for i in range(len(hidden_dims))])
                
        if last_norm:
            self.last_norm = nn.LayerNorm(normalized_shape=[hidden_dims[-1], 4, 4, 4], eps=1e-6)
        else: 
            self.last_norm = nn.Identity()
                
    def _make_layer(self, block, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer=norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.layers(x)
        x = self.last_norm(x)
        return F.adaptive_max_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, feature_dim]
