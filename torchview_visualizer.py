#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Torchview绘制神经网络架构图
专门为CSPEncoder3D和相关模型设计的可视化工具
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Optional, Tuple, List
from collections import OrderedDict

# 添加项目路径
sys.path.append('.')

try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError:
    print("警告: torchview未安装，请运行: pip install torchview")
    TORCHVIEW_AVAILABLE = False

# 导入模型
from src.model.encoder.cubenet_rtm import (
    CSPEncoder3D, CSPNeXtBlock3D, CSPBlock3D, 
    ChannelAttention3D, CSPPAFPN3D, DepthwiseSeparableConv3d, SPP3D
)
from src.utils.tools import get_obj_from_str

class TorchviewVisualizer:
    def visualize_rtm_encoder_3d(self, 
                                input_shape: Tuple[int, ...] = (2, 32, 32, 32, 32),
                                save_path: Optional[str] = None,
                                show_shapes: bool = True,
                                show_layer_names: bool = True):
        """可视化CSPEncoder3D架构"""
        print("正在创建CSPEncoder3D模型...")
        
        class NamedCSPEncoder3D(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.stem = base_model.stem
                self.stages = base_model.stages
                self.neck = base_model.neck
                self.global_pool = base_model.global_pool
            
            def forward(self, x):
                x = self.stem(x)
                features = []
                for i, stage in enumerate(self.stages):
                    x = stage(x)
                    if i >= 1:
                        features.append(x)
                neck_outs = self.neck(features)
                x = neck_outs[-1]
                global_feat = self.global_pool(x).flatten(1)
                return global_feat
        
        base_model = CSPEncoder3D(
            in_channels=32,
            base_channels=64,
            stage_channels=[128, 256, 512, 1024],
            stage_blocks=[2, 4, 4, 2],
            expandsion=0.5,
            channel_attention=True,
            norm_layer='torch.nn.GroupNorm'
        )
        model = NamedCSPEncoder3D(base_model)
        model.eval()
        
        input_tensor = torch.randn(*input_shape)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成架构图...")
        
        model_graph = draw_graph(
            model, 
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "rtm_encoder_3d_architecture",
            directory="./",
            expand_nested=True,
            depth=2,
            graph_name="CSPEncoder3D Architecture"
        )
        
        if save_path:
            print(f"CSPEncoder3D架构图已保存到: {save_path}")
        return model_graph

    def visualize_depthwise_separable_conv3d(self,
                                            in_channels: int = 128,
                                            out_channels: int = 256,
                                            kernel_size: int = 3,
                                            save_path: Optional[str] = None):
        """可视化DepthwiseSeparableConv3d模块"""
        print("正在创建DepthwiseSeparableConv3d模型...")
        
        class NamedDepthwiseSeparableConv3d(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.depthwise = base_model.depthwise
                self.pointwise = base_model.pointwise
            
            def forward(self, x):
                x = self.depthwise(x)
                x = self.pointwise(x)
                return x
        
        base_model = DepthwiseSeparableConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm_layer=nn.GroupNorm
        )
        model = NamedDepthwiseSeparableConv3d(base_model)
        model.eval()
        
        input_tensor = torch.randn(2, in_channels, 8, 8, 8)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成DepthwiseSeparableConv3d架构图...")
        
        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "depthwise_separable_conv3d_architecture",
            directory="./",
            expand_nested=True,
            depth=2,
            graph_name="DepthwiseSeparableConv3d Architecture"
        )
        
        if save_path:
            print(f"DepthwiseSeparableConv3d架构图已保存到: {save_path}")
        return model_graph

    def visualize_cspnext_block(self,
                               in_channels: int = 256,
                               out_channels: int = 256,
                               kernel_size: int = 5,
                               save_path: Optional[str] = None):
        """可视化CSPNeXtBlock3D详细结构"""
        print("正在创建CSPNeXtBlock3D模型...")
        
        class NamedCSPNeXtBlock3D(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.conv1 = base_model.conv1
                self.conv2 = base_model.conv2
                self.add_identity = base_model.add_identity
            
            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.conv2(out)
                if self.add_identity:
                    return out + identity
                return out
        
        base_model = CSPNeXtBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            expandsion=0.5,
            add_identity=True,
            use_depthwise=True,
            kernel_size=kernel_size,
            norm_layer=nn.GroupNorm
        )
        model = NamedCSPNeXtBlock3D(base_model)
        model.eval()
        
        input_tensor = torch.randn(2, in_channels, 8, 8, 8)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成CSPNeXtBlock3D架构图...")
        
        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "cspnext_block_architecture",
            directory="./",
            expand_nested=True,
            depth=2,
            graph_name="CSPNeXtBlock3D Architecture"
        )
        
        if save_path:
            print(f"CSPNeXtBlock3D架构图已保存到: {save_path}")
        return model_graph

    def visualize_csp_block(self,
                           in_channels: int = 256,
                           out_channels: int = 256,
                           num_blocks: int = 4,
                           save_path: Optional[str] = None):
        """可视化CSP Block详细结构"""
        print("正在创建CSP Block模型...")
        
        class NamedCSPBlock3D(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.main_conv = base_model.main_conv
                self.short_conv = base_model.short_conv
                self.final_conv = base_model.final_conv
                self.blocks = base_model.blocks
                self.channel_attention = base_model.channel_attention
                if self.channel_attention:
                    self.attention = base_model.attention
            
            def forward(self, x):
                x_short = self.short_conv(x)
                x_main = self.main_conv(x)
                x_main = self.blocks(x_main)
                x_final = torch.cat((x_main, x_short), dim=1)
                if self.channel_attention:
                    x_final = self.attention(x_final)
                return self.final_conv(x_final)
        
        base_model = CSPBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            expandsion=0.5,
            num_blocks=num_blocks,
            add_identity=True,
            use_depthwise=True,
            use_cspnext_block=True,
            channel_attention=True,
            norm_layer=nn.GroupNorm
        )
        model = NamedCSPBlock3D(base_model)
        model.eval()
        
        input_tensor = torch.randn(2, in_channels, 8, 8, 8)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成CSP Block架构图...")
        
        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "csp_block_architecture",
            directory="./",
            expand_nested=True,
            depth=2,
            graph_name="CSP Block 3D Architecture"
        )
        
        if save_path:
            print(f"CSP Block架构图已保存到: {save_path}")
        return model_graph

    def visualize_channel_attention(self,
                                   channels: int = 256,
                                   save_path: Optional[str] = None):
        """可视化Channel Attention模块"""
        print("正在创建Channel Attention模型...")
        
        class NamedChannelAttention3D(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.avg_pool = base_model.avg_pool
                self.fc = base_model.fc
            
            def forward(self, x):
                att = self.avg_pool(x)
                att = self.fc(att)
                return x * att
        
        base_model = ChannelAttention3D(channels=channels, reduction_ratio=16)
        model = NamedChannelAttention3D(base_model)
        model.eval()
        
        input_tensor = torch.randn(2, channels, 8, 8, 8)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成Channel Attention架构图...")
        
        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "channel_attention_architecture",
            directory="./",
            expand_nested=True,
            depth=2,
            graph_name="Channel Attention 3D Architecture"
        )
        
        if save_path:
            print(f"Channel Attention架构图已保存到: {save_path}")
        return model_graph

    def visualize_csppafpn_3d(self,
                             in_channels=[256, 512, 1024],
                             save_path: Optional[str] = None):
        """可视化CSPPAFPN3D架构"""
        print("正在创建CSPPAFPN3D模型...")
        
        class NamedCSPPAFPN3D(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.spp = base_model.spp
                self.upsample = base_model.upsample
                self.reduce_layers = base_model.reduce_layers
                self.top_down_blocks = base_model.top_down_blocks
                self.downsample_layers = base_model.downsample_layers
                self.trans_layers = base_model.trans_layers
                self.bottom_up_blocks = base_model.bottom_up_blocks
                self.out_indices = base_model.out_indices
            
            def forward(self, x):
                # 将输入按通道和空间尺寸拆分
                B = x.shape[0]
                
                # 创建三个不同尺寸的特征图
                inputs = [
                    x[:, :256, :32, :32, :32],                    # P3 [B, 256, 32, 32, 32]
                    x[:, 256:768, :16, :16, :16],                 # P4 [B, 512, 16, 16, 16]
                    x[:, 768:, :8, :8, :8]                        # P5 [B, 1024, 8, 8, 8]
                ]
                
                inputs[-1] = self.spp(inputs[-1])
                
                feat = inputs[-1]
                top_down_feats = [feat]
                
                for idx in range(len(inputs) - 1):
                    feat = self.reduce_layers[idx](feat)
                    feat = self.upsample(feat)
                    feat = torch.cat([feat, inputs[-(idx+2)]], dim=1)
                    feat = self.top_down_blocks[idx](feat)
                    top_down_feats.append(feat)
                
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
        
        base_model = CSPPAFPN3D(
            in_channels=in_channels,
            out_channels=None,
            out_indices=(1, 2),
            num_csp_blocks=2,
            expandsion=0.5,
            norm_layer=nn.GroupNorm
        )
        model = NamedCSPPAFPN3D(base_model)
        model.eval()
        
        input_tensor = torch.randn(2, sum(in_channels), 32, 32, 32)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成CSPPAFPN3D架构图...")
        
        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "csppafpn_3d_architecture",
            directory="./",
            expand_nested=True,
            depth=1,
            graph_name="CSPPAFPN3D Architecture"
        )
        
        if save_path:
            print(f"CSPPAFPN3D架构图已保存到: {save_path}")
        return model_graph

    def visualize_spp_3d(self,
                         in_channels: int = 256,
                         out_channels: int = 256,
                         save_path: Optional[str] = None):
        """可视化SPP3D模块"""
        print("正在创建SPP3D模型...")
        
        class NamedSPP3D(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.conv1 = base_model.conv1
                self.pools = base_model.pools
                self.conv2 = base_model.conv2
            
            def forward(self, x):
                x = self.conv1(x)
                pool_outs = [x] + [pool(x) for pool in self.pools]
                x = torch.cat(pool_outs, dim=1)
                return self.conv2(x)
        
        base_model = SPP3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=(5, 9, 13),
            norm_layer=nn.GroupNorm
        )
        model = NamedSPP3D(base_model)
        model.eval()
        
        input_tensor = torch.randn(2, in_channels, 32, 32, 32)
        print(f"输入形状: {input_tensor.shape}")
        print("正在生成SPP3D架构图...")
        
        model_graph = draw_graph(
            model,
            input_data=input_tensor,
            save_graph=bool(save_path),
            filename=save_path or "spp_3d_architecture",
            directory="./",
            expand_nested=True,
            depth=2,
            graph_name="SPP3D Architecture"
        )
        
        if save_path:
            print(f"SPP3D架构图已保存到: {save_path}")
        return model_graph

def demo_all_visualizations():
    """演示所有可视化功能"""
    print("=== Torchview神经网络架构可视化演示 ===")
    
    if not TORCHVIEW_AVAILABLE:
        print("错误: 请先安装torchview")
        print("运行命令: pip install torchview")
        return
    
    # 创建results目录
    os.makedirs("results", exist_ok=True)
    
    # 创建可视化器
    visualizer = TorchviewVisualizer()
    
    # # 1. 可视化CSPEncoder3D
    # print("\n1. 可视化CSPEncoder3D架构...")
    # visualizer.visualize_rtm_encoder_3d(
    #     input_shape=(2, 32, 32, 32, 32),
    #     save_path="results/rtm_encoder_3d_torchview",
    #     show_shapes=True,
    #     show_layer_names=True
    # )
    
    # # 2. 可视化DepthwiseSeparableConv3d
    # print("\n2. 可视化DepthwiseSeparableConv3d模块...")
    # visualizer.visualize_depthwise_separable_conv3d(
    #     in_channels=128,
    #     out_channels=256,
    #     kernel_size=3,
    #     save_path="results/depthwise_separable_conv3d_torchview"
    # )
    
    # # 3. 可视化CSPNeXtBlock3D
    # print("\n3. 可视化CSPNeXtBlock3D详细结构...")
    # visualizer.visualize_cspnext_block(
    #     in_channels=256,
    #     out_channels=256,
    #     kernel_size=5,
    #     save_path="results/cspnext_block_torchview"
    # )
    
    # # 4. 可视化CSP Block
    # print("\n4. 可视化CSP Block详细结构...")
    # visualizer.visualize_csp_block(
    #     in_channels=256,
    #     out_channels=256,
    #     num_blocks=4,
    #     save_path="results/csp_block_torchview"
    # )
    
    # # 5. 可视化Channel Attention
    # print("\n5. 可视化Channel Attention模块...")
    # visualizer.visualize_channel_attention(
    #     channels=256,
    #     save_path="results/channel_attention_torchview"
    # )
    
    # 6. 可视化CSPPAFPN3D
    print("\n6. 可视化CSPPAFPN3D模块...")
    visualizer.visualize_csppafpn_3d(
        in_channels=[256, 512, 1024],
        save_path="results/csppafpn_3d_torchview"
    )
    
    # 7. 可视化SPP3D
    print("\n7. 可视化SPP3D模块...")
    visualizer.visualize_spp_3d(
        in_channels=256,
        out_channels=256,
        save_path="results/spp_3d_torchview"
    )
    
    print("\n=== 所有架构图已生成完成！ ===")
    print("生成的文件:")
    print("- results/rtm_encoder_3d_torchview.png: CSPEncoder3D整体架构")
    print("- results/depthwise_separable_conv3d_torchview.png: DepthwiseSeparableConv3d模块")
    print("- results/cspnext_block_torchview.png: CSPNeXtBlock3D详细结构")
    print("- results/csp_block_torchview.png: CSP Block详细结构")
    print("- results/channel_attention_torchview.png: Channel Attention模块")
    print("- results/csppafpn_3d_torchview.png: CSPPAFPN3D模块")
    print("- results/spp_3d_torchview.png: SPP3D模块")


def install_torchview():
    """安装torchview的辅助函数"""
    import subprocess
    import sys
    
    print("正在安装torchview...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchview"])
        print("torchview安装成功！")
        return True
    except subprocess.CalledProcessError:
        print("torchview安装失败，请手动安装: pip install torchview")
        return False


if __name__ == "__main__":
    # 检查并安装torchview
    if not TORCHVIEW_AVAILABLE:
        if install_torchview():
            # 重新导入
            try:
                from torchview import draw_graph
                TORCHVIEW_AVAILABLE = True
            except ImportError:
                print("重新导入失败，请重启Python环境")
                exit(1)
        else:
            exit(1)
    
    # 运行演示
    demo_all_visualizations()