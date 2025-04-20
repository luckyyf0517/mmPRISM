import os
import time
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from configs.radar import iwr1843 as radar_cfg


class Simulation(): 
    def __init__(self, dtype=torch.float64, ctype=torch.complex64): 
        self.radar_cfg = radar_cfg
        self.simulator = mmSimulator(radar_cfg, dtype=dtype, ctype=ctype)
        self.simulator.init()
        self.window = torch.hann_window(radar_cfg.num_adc_samples, dtype=dtype)
        
        self.start_freq = radar_cfg.start_freq
        self.freq_slope = radar_cfg.freq_slope
        self.adc_sample_rate = radar_cfg.adc_sample_rate
        self.adc_start_time = radar_cfg.adc_start_time
        self.num_adc_samples = radar_cfg.num_adc_samples
        self.ts = self.adc_start_time + torch.arange(self.num_adc_samples, dtype=dtype) / self.adc_sample_rate 
        self.fs = self.ts * self.freq_slope 
        self.chirp_period = radar_cfg.chirp_period
        self.num_chirps = radar_cfg.num_chirps
        self.time_steps = torch.arange(self.num_chirps, dtype=dtype) * self.chirp_period
        self.ramp_end_time = radar_cfg.ramp_end_time
        
        self.dtype = dtype
        self.ctype = ctype
        
    def forward(self, points_3d, velocities_3d):
        """
        Forward pass using 3D points and velocities
        
        Args:
            points_3d: [N, 3] 3D points in camera coordinates
            velocities_3d: [N, 3] 3D velocities in m/s
            
        Returns:
            radar_signal: [doppler, range] radar signal
        """
        path_dict = self.simulator.compute_paths_from_points(
            points_3d=points_3d,
            velocities_3d=velocities_3d
        )
        radar_frame = self.get_raw_radar_frame(path_dict)
        doppler = self.get_signal(radar_frame)
        return doppler
    
    def get_raw_radar_frame(self, path_dict, save_cuda_memory=False):
        """
        Generate raw radar frame from path information
        
        Args:
            path_dict: Dictionary containing path information with keys:
                'a': [num_rx, max_num_paths] amplitude 
                'tau': [num_rx, max_num_paths] time delay
                'vel': [max_num_paths] velocity
            save_cuda_memory: Whether to use memory-efficient computation
            
        Returns:
            radar_frame: [num_chirps, num_rx, num_samples] complex radar signals
        """
        a = path_dict['a']  # [num_rx, max_num_paths]
        tau = path_dict['tau']  # [num_rx, max_num_paths]
        vel = path_dict['vel']  # [max_num_paths]
        tau_velocity = vel * 2 / 2.99792458e8  # [max_num_paths]
        time_steps = self.time_steps[:, None, None]  # [num_chirps, 1]
        # [num_chirps, num_rx, 1, max_num_paths]
        tau_chirp = tau_velocity * time_steps.to(a.device)  
        tau_chirp = tau.unsqueeze(0) + tau_chirp
        tau_chirp = tau_chirp[:, :, None, :] 
        # [1, 1, num_samples, 1]
        frequencies = self.fs[None, None, :, None].to(a.device)
        # [num_chirps, num_rx, num_samples, max_num_paths]
        ft_phase = 2 * np.pi * (frequencies + self.start_freq) * tau_chirp
        ft_phase %= (2 * np.pi)
        # [1, num_rx, 1, max_num_paths]
        a = a[None, :, None, :]
        # [num_chirps, num_rx, num_samples]
        radar_frame = (a * torch.exp(1j * ft_phase)).sum(dim=-1)
        return radar_frame.to(self.ctype)
    
    def get_signal(self, raw_radar_frame):
        """
        Convert radar frame to mmwave cube
        """
        # num_chirps, num_antenna, num_samples = radar_frame.shape
        radar_frame = raw_radar_frame.clone()
        radar_frame = radar_frame * torch.hann_window(radar_frame.shape[2], device=radar_frame.device)
        radar_frame = torch.fft.fft(radar_frame, dim=2)
        radar_frame = radar_frame[:, :, :32]
        radar_frame = radar_frame * torch.hann_window(radar_frame.shape[0], device=radar_frame.device)[:, None, None]
        radar_frame = torch.fft.fftshift(torch.fft.fft(radar_frame, dim=0), dim=0)
        radar_frame = radar_frame.abs() ** 2
        radar_frame = radar_frame.sum(dim=-1)
        return radar_frame # [num_chirps, num_rx]


class mmSimulator(nn.Module): 
    def __init__(self, radar_cfg, dtype=torch.float32, ctype=torch.complex64): 
        super().__init__()
        # Load integrated scene
        self.frequency = radar_cfg.start_freq
        # Equivalent to 3tx and 4rx
        self.num_tx = 1
        self.num_rx = 12  # 修改为12个接收天线

        self.dtype = dtype
        self.ctype = ctype

    def init(self):
        """
        Initialize the simulator with camera parameters
        """
        self.radar_position = np.array([0.0, 0.0, 0.0])  # 更新雷达位置
        self.camera_position = self.radar_position + np.array([0.0, 0.0, 0.0])
        
        rx_ref_array = np.array([
            [-1.414, 0.0, 0.0],  # -90deg
            [-1.0, 0.0, -1.0],  # -45deg
            [0.0, 0.0, -1.414],  # 0deg
            [1.0, 0.0, -1.0],  # 45deg
            [1.414, 0.0, 0.0],  # 90deg
        ])
        
        self.register_buffer('tx_position', torch.tensor(self.radar_position), persistent=False)
        self.register_buffer('rx_positions', torch.tensor(self.radar_position[None] + rx_ref_array), persistent=False)
        self.scat_pattern = None
        self.radar_pattern = None
        
    def compute_paths_from_points(self, points_3d, velocities_3d):
        """
        Compute path delay using 3D points. No physics considered.
        
        Args:
            points_3d: [N, 3] 3D points in camera coordinates
            velocities_3d: [N, 3] 3D velocities in m/s
            
        Returns:
            dict with keys:
                a: [num_rx, max_num_paths] 
                tau: [num_rx, max_num_paths]
                vel: [num_rx, max_num_paths]
        """
        # 计算发射路径
        k_i = points_3d - self.tx_position  # [N, 3]
        k_s = (self.rx_positions[:, None] - points_3d)  # [num_rx, N, 3]
        
        # 计算距离
        k_i_length = torch.norm(k_i, dim=-1)  # [N]
        k_s_length = torch.norm(k_s, dim=-1)  # [num_rx, N]
        distances = k_i_length[None] + k_s_length  # [num_rx, N]
        
        tau = distances / 2.99792458e8  # [num_rx, N]
        a = torch.ones_like(tau) / (distances / 2) ** 2
        
        # 计算径向速度
        k_i = k_i / k_i_length.unsqueeze(-1)  # [N, 3]
        k_s = k_s / k_s_length.unsqueeze(-1)  # [num_rx, N, 3]
        vel = (torch.sum(velocities_3d * k_i, dim=-1)[None] - 
               torch.sum(velocities_3d * k_s, dim=-1)) / 2  # [num_rx, N]
        
        # from IPython import embed; embed(header='compute_paths_from_points')
        
        return {
            'a': a.to(self.dtype),
            'tau': tau.to(self.dtype),
            'vel': vel.to(self.dtype)
        }
