import os
import time
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from config.radar import iwr1843 as radar_cfg
from src.fmcw.beamformer import build_steering_vector, build_steering_vector_1d

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Simulation(nn.Module): 
    def __init__(self, dtype=torch.float32, ctype=torch.complex64): 
        super().__init__()
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
        
        self.D, self.R, self.W, self.H = 32, 32, 32, 32
        azi_ele_id = torch.tensor(self.simulator.D)
        azi_theta_grid = torch.linspace(-np.pi/6, np.pi/6, self.W)
        ele_theta_grid = torch.linspace(-np.pi/6, np.pi/6, self.H)
        self.bm_weights = build_steering_vector(azi_ele_id, azi_theta_grid, ele_theta_grid)
        self.bm_weights = nn.Parameter(self.bm_weights, requires_grad=False)
        
        self.dtype = dtype
        self.ctype = ctype

    def forward(self, points_3d, velocities_3d):
        """
        Forward pass using 3D points and velocities
        
        Args:
            points_3d: [B, N, 3] 3D points in camera coordinates for batch
            velocities_3d: [B, N, 3] 3D velocities in m/s for batch
            
        Returns:
            radar_signal: [B, doppler, range] radar signal
        """
        batch_size = points_3d.shape[0]
        radar_signal_list = []
        
        points_3d = process_point_cloud(points_3d)
        velocities_3d = process_point_cloud(velocities_3d)
        
        for b in range(batch_size):
            path_dict = self.simulator.compute_paths_from_points(
                points_3d=points_3d[b],
                velocities_3d=velocities_3d[b])
            radar_frame = self.get_raw_radar_frame(path_dict)
            radar_signal = self.get_signal(radar_frame)
            radar_signal_list.append(radar_signal)
            
        radar_signal_batch = torch.stack(radar_signal_list, dim=0)
        return radar_signal_batch
    
    def get_raw_radar_frame(self, path_dict, save_cuda_memory=True):
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
        if save_cuda_memory: 
            a = a.to(torch.float16)
            ft_phase = ft_phase.to(torch.float16)
        radar_frame = (a * torch.exp(1j * ft_phase)).sum(dim=-1)
        return radar_frame.to(self.dtype)
    
    def get_signal(self, raw_radar_frame):
        """
        Convert radar frame to mmwave cube
        """
        radar_frame = raw_radar_frame.clone()
        num_chirps, num_antenna, num_samples = radar_frame.shape
        # range fft
        radar_frame = radar_frame * torch.hann_window(num_samples, device=radar_frame.device)
        radar_frame = torch.fft.fft(radar_frame, dim=2)
        radar_frame = radar_frame[:, :, :self.R]
        # doppler fft
        radar_frame = radar_frame * torch.hann_window(num_chirps, device=radar_frame.device)[:, None, None]
        radar_frame = torch.fft.fftshift(torch.fft.fft(radar_frame, dim=0), dim=0)
        # beamforming
        radar_frame = torch.einsum('cd,acb->abd', self.bm_weights, radar_frame)
        radar_frame = radar_frame.abs() ** 2
        return radar_frame.reshape(self.D, self.R, self.W, self.H)


class mmSimulator(nn.Module): 
    def __init__(self, radar_cfg, dtype=torch.float32, ctype=torch.complex64): 
        super().__init__()
        # Load integrated scene
        self.frequency = radar_cfg.start_freq
        self.dtype = dtype
        self.ctype = ctype

    def init(self):
        """
        Initialize the simulator with camera parameters
        """
        self.radar_position = np.array([0.0, 0.0, -0.80])  # 更新雷达位置
        self.camera_position = self.radar_position + np.array([0.0, 0.0, 0.0])
        
        self.D = np.array(get_index()) - np.array([43, 0])
        d = (2.99792458e8 / self.frequency) / 2     # wave_length / 2
        rx_ref_array = np.column_stack((self.D[:, 0], self.D[:, 1], np.zeros(len(self.D))))
        rx_ref_array = rx_ref_array * d
        
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


def get_index(return_index=False):
    D = [
        [ 0,  0], [ 1,  0], [ 2,  0], [ 3,  0], [11,  0], [12,  0], [13,  0], [14,  0],
        [46,  0], [47,  0], [48,  0], [49,  0], [50,  0], [51,  0], [52,  0], [53,  0],
        [ 4,  0], [ 5,  0], [ 6,  0], [ 7,  0], [15,  0], [16,  0], [17,  0], [18,  0],
        [50,  0], [51,  0], [52,  0], [53,  0], [54,  0], [55,  0], [56,  0], [57,  0],
        [ 8,  0], [ 9,  0], [10,  0], [11,  0], [19,  0], [20,  0], [21,  0], [22,  0],
        [54,  0], [55,  0], [56,  0], [57,  0], [58,  0], [59,  0], [60,  0], [61,  0],
        [12,  0], [13,  0], [14,  0], [15,  0], [23,  0], [24,  0], [25,  0], [26,  0],
        [58,  0], [59,  0], [60,  0], [61,  0], [62,  0], [63,  0], [64,  0], [65,  0],
        [16,  0], [17,  0], [18,  0], [19,  0], [27,  0], [28,  0], [29,  0], [30,  0],
        [62,  0], [63,  0], [64,  0], [65,  0], [66,  0], [67,  0], [68,  0], [69,  0],
        [20,  0], [21,  0], [22,  0], [23,  0], [31,  0], [32,  0], [33,  0], [34,  0],
        [66,  0], [67,  0], [68,  0], [69,  0], [70,  0], [71,  0], [72,  0], [73,  0],
        [24,  0], [25,  0], [26,  0], [27,  0], [35,  0], [36,  0], [37,  0], [38,  0],
        [70,  0], [71,  0], [72,  0], [73,  0], [74,  0], [75,  0], [76,  0], [77,  0],
        [28,  0], [29,  0], [30,  0], [31,  0], [39,  0], [40,  0], [41,  0], [42,  0],
        [74,  0], [75,  0], [76,  0], [77,  0], [78,  0], [79,  0], [80,  0], [81,  0],
        [32,  0], [33,  0], [34,  0], [35,  0], [43,  0], [44,  0], [45,  0], [46,  0],
        [78,  0], [79,  0], [80,  0], [81,  0], [82,  0], [83,  0], [84,  0], [85,  0],
        [ 9,  1], [10,  1], [11,  1], [12,  1], [20,  1], [21,  1], [22,  1], [23,  1],
        [55,  1], [56,  1], [57,  1], [58,  1], [59,  1], [60,  1], [61,  1], [62,  1],
        [10,  4], [11,  4], [12,  4], [13,  4], [21,  4], [22,  4], [23,  4], [24,  4],
        [56,  4], [57,  4], [58,  4], [59,  4], [60,  4], [61,  4], [62,  4], [63,  4],
        [11,  6], [12,  6], [13,  6], [14,  6], [22,  6], [23,  6], [24,  6], [25,  6],
        [57,  6], [58,  6], [59,  6], [60,  6], [61,  6], [62,  6], [63,  6], [64,  6]
        ]
    D_uni = np.unique(D, axis=0).tolist()
    
    to_be_removed = [
        [13, 6], [14, 6], [24, 6], [25, 6], [63, 6], [64, 6], 
        [10, 4], [13, 4], [21, 4], [24, 4], [56, 4], [63, 4], 
        [9, 1], [10, 1], [20, 1], [21, 1], [55, 1], [56, 1]]
    for item in to_be_removed: 
        D_uni.remove(item)
    
    index = []
    for cor in D_uni:
        index.append(D.index(cor))
    if return_index: 
        return index
    else: 
        return D_uni


def process_point_cloud(data):
    """
    Process a point cloud data of shape [T, N, 3] and return a new point cloud of shape [T, N', 3].
    
    Args:
        data: torch.Tensor of shape [T, N, 3] representing the point cloud data.
        
    Returns:
        torch.Tensor of shape [T, N', 3] representing the processed point cloud.
    """
    # Extract body and hand points
    body = data[:, 5:11, :]
    handl = data[:, -42:-21, :]
    handr = data[:, -21:, :]

    # Define body skeleton
    body_skeleton = torch.tensor([
        [0, 2], [2, 4],
        [1, 3], [3, 5],
        [0, 1]
    ], device=data.device)

    # Vectorized interpolation of points
    def interpolate_points_vectorized(p1, p2, num_points=3):
        t_values = torch.linspace(0, 1, num_points + 2, device=data.device)[1:-1]
        t_values = t_values.unsqueeze(0).unsqueeze(-1)  # Reshape t_values for broadcasting
        return p1.unsqueeze(1) + (p2 - p1).unsqueeze(1) * t_values

    # Vectorized addition of body points and interpolated points
    interpolated_points_list = []
    for i, j in body_skeleton:
        interpolated_points = interpolate_points_vectorized(body[:, i], body[:, j])
        interpolated_points_list.append(interpolated_points)
    interpolated_points = torch.cat(interpolated_points_list, dim=1)
    all_body_points = torch.cat((body, interpolated_points, handl, handr), dim=1)
    all_body_points[..., 2] *= 0.6
    all_body_points[..., 1] -= 0.15
    return all_body_points
