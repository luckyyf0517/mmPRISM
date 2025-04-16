import os
import time
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import sys; sys.path.append('.')

from src.fmcw.fmcw_radar import FMCWRadar
from configs.radar import iwr1843 as radar_cfg


class Simulation(): 
    def __init__(self, dtype=torch.float64, ctype=torch.complex64): 
        self.radar_cfg = radar_cfg
        self.simulator = mmSimulator(radar_cfg, dtype=dtype, ctype=ctype)
        self.simulator.init(
            image_width=160,
            image_height=120,
            fx=1500,
            fy=1500,
            cx=80,
            cy=60
        )
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
        
        if self.simulator.TX is not None: 
            self.TX = torch.tensor(self.simulator.TX)
        else: 
            self.TX = None
        
    def forward(self, depth_map, velocity_map=None): 
        """
        Forward pass of the simulation using depth map input
        
        Args:
            depth_map: [B, H, W] depth values in meters
            velocity_map: [B, H, W, 3] velocity values in m/s (optional)
            
        Returns:
            path_info: [3, B, num_rx, max_num_paths] concatenated path information
                      containing amplitude, time delay, and velocity
        """
        path_dict = self.simulator.compute_paths_using_depth(
            depth_map=depth_map,
            velocity_map=velocity_map,
            consider_energy=True,
            remove_unvisible=True
        )
        # 将三个分量拼接在一起，保持与原来相同的输出格式
        return torch.stack([
            path_dict['a'],      # [B, num_rx, max_num_paths]
            path_dict['tau'],    # [B, num_rx, max_num_paths]
            path_dict['vel']     # [B, num_rx, max_num_paths]
        ], dim=0)  # [3, B, num_rx, max_num_paths]

    def get_raw_radar_frame(self, path_info, TX=None, save_cuda_memory=False): 
        """
        Generate raw radar frame from path information
        
        Args:
            path_info: [3, B, num_rx, max_num_paths] path information
            TX: Optional TX antenna indices
            save_cuda_memory: Whether to use memory-efficient computation
            
        Returns:
            radar_frame: [B, num_chirps, num_rx, num_samples] complex radar signals
        """
        # (1, num_ant, num_path)
        dtype = path_info.dtype
        a, tau, vel = path_info.chunk(3, dim=0)  # Each [1, B, num_rx, max_num_paths]
        tau_velocity = vel * 2 / 2.99792458e8
        
        # fullfill the batch using chirps
        if TX is not None: 
            time_steps_delay = TX * self.ramp_end_time
            time_steps = self.time_steps[:, None, None] + time_steps_delay[None, :, None]
        else: 
            time_steps = self.time_steps[:, None, None]
            
        tau_chirp = tau + tau_velocity * time_steps.to(path_info.device)
        a = a.unsqueeze(-2)
        tau_chirp = tau_chirp.unsqueeze(-2)
        frequencies = self.fs.reshape(1, 1, -1, 1).to(path_info.device)
        ft_phase = 2 * np.pi * (frequencies + self.start_freq) * tau_chirp
        ft_phase %= (2 * np.pi)
        
        if save_cuda_memory: 
            radar_frame = torch.zeros(ft_phase.shape[:3], dtype=torch.complex64, device=path_info.device)
            for i in range(radar_frame.shape[1]): 
                radar_frame[:, i] = (a[:, i] * torch.exp(1j * ft_phase[:, i])).sum(dim=-1)
        else: 
            radar_frame = (a * torch.exp(1j * ft_phase)).sum(dim=-1)
        return radar_frame


class mmSimulator(nn.Module): 
    def __init__(self, radar_cfg, dtype=torch.float32, ctype=torch.complex64): 
        super().__init__()
        # Load integrated scene
        self.radar = FMCWRadar(radar_cfg, dtype=dtype, ctype=ctype)
        self.frequency = radar_cfg.start_freq
        # Equivalent to 3tx and 4rx
        self.num_tx = 1
        self.num_rx = 12
        
        # Camera parameters with default values
        self.fx = 1500
        self.fy = 1500
        self.cx = 80
        self.cy = 60
        self.image_width = 160
        self.image_height = 120
        
        self.dtype = dtype
        self.ctype = ctype

    def init(self, image_width=160, image_height=120, fx=1500, fy=1500, cx=80, cy=60):
        """
        Initialize the simulator with camera parameters
        
        Args:
            image_width: Width of the depth image
            image_height: Height of the depth image
            fx, fy: Focal lengths
            cx, cy: Principal point
        """
        self.image_width = image_width
        self.image_height = image_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.radar_position = np.array([-0.05, 0.0, -0.06])  
        self.camera_position = self.radar_position + np.array([0.0, 0.0, 0.0])
        d = (2.99792458e8 / self.frequency) / 2     # wave_length / 2
        
        # 192 * rx large array for mmwave cascade radar
        rx_ref_array, _, TX = get_index()
        rx_ref_array = np.array(rx_ref_array) #- np.array([43, 0]) # move to the center
        rx_ref_array = np.column_stack((rx_ref_array[:, 0], rx_ref_array[:, 1], np.zeros(len(rx_ref_array))))
        rx_ref_array = rx_ref_array * d
        self.TX = TX
        
        self.register_buffer('tx_position', torch.tensor(self.radar_position), persistent=False)
        self.register_buffer('rx_positions', torch.tensor(self.radar_position[None] + rx_ref_array), persistent=False)
        self.scat_pattern = None
        self.radar_pattern = None
        
    def depth_to_points(self, depth_map):
        """
        Convert depth map to 3D points in camera coordinates using direct computation
        
        Args:
            depth_map: [B, H, W] depth values in meters
            
        Returns:
            points: [B, N, 3] 3D points in camera coordinates
            valid_mask: [B, H*W] boolean mask for valid points
        """
        batch_size, height, width = depth_map.shape
        
        # Reshape depth map
        depths = depth_map.reshape(batch_size, -1)  # [B, H*W]
        
        # Create indices for each pixel
        v = torch.arange(height, device=depth_map.device)
        u = torch.arange(width, device=depth_map.device)
        
        # Compute x, y, z coordinates for all pixels
        x = ((u.view(1, 1, -1) - self.cx) / self.fx)  # [1, 1, W]
        y = ((v.view(1, -1, 1) - self.cy) / self.fy)  # [1, H, 1]
        
        # Expand for batch processing
        x = x.expand(batch_size, height, width)  # [B, H, W]
        y = y.expand(batch_size, height, width)  # [B, H, W]
        
        # Reshape to match depths
        x = x.reshape(batch_size, -1)  # [B, H*W]
        y = y.reshape(batch_size, -1)  # [B, H*W]
        
        # Compute 3D coordinates
        points_x = x * depths  # [B, H*W]
        points_y = y * depths  # [B, H*W]
        points_z = depths      # [B, H*W]
        
        # Stack coordinates
        points_3d = torch.stack([points_x, points_y, points_z], dim=-1)  # [B, H*W, 3]
        
        # Create valid mask
        valid_mask = depths > 0  # [B, H*W]
        
        return points_3d, valid_mask
        
    def compute_paths_using_depth(self, depth_map, velocity_map=None, consider_energy=True, remove_unvisible=True): 
        """
        Compute path delay using depth map. No physics considered.  
        
        Args:
            depth_map: [B, H, W] depth values in meters
            velocity_map: [B, H, W, 3] velocity values in m/s (optional)
            consider_energy: Whether to consider energy loss
            remove_unvisible: Whether to remove points not visible from radar
            
        Returns:
            dict with keys:
                a: [B, num_rx, max_num_paths]
                tau: [B, num_rx, max_num_paths]
                vel: [B, num_rx, max_num_paths]
        """
        # Convert depth to 3D points
        points_3d, valid_mask = self.depth_to_points(depth_map)  # [B, H*W, 3], [B, H*W]
        batch_size = depth_map.shape[0]
        
        # Convert velocity map to 3D velocities if provided
        if velocity_map is not None:
            velocity_map = velocity_map.reshape(batch_size, -1, 3)  # [B, H*W, 3]
            velocity = velocity_map[valid_mask]  # [B, N, 3]
        else:
            velocity = torch.zeros_like(points_3d)  # [B, H*W, 3]
        
        # Calculate normals for each point (simplified approach)
        normals = torch.zeros_like(points_3d)  # [B, H*W, 3]
        normals[..., 2] = 1.0  # Assuming normals point toward camera
        
        # Calculate areas (simplified - each point represents a pixel)
        area = torch.ones(batch_size, points_3d.shape[1], device=points_3d.device)  # [B, H*W]
        area = area * valid_mask.float()
        
        if remove_unvisible: 
            view_direction = self.tx_position.unsqueeze(0).unsqueeze(0) - points_3d  # [B, H*W, 3]
            dot_product = torch.sum(normals * view_direction, dim=-1)  # [B, H*W]
            visible_mask = (dot_product > 0) & valid_mask  # [B, H*W]
            points_3d = points_3d[visible_mask]  # [N_valid, 3]
            normals = normals[visible_mask]  # [N_valid, 3]
            area = area[visible_mask]  # [N_valid]
            velocity = velocity[visible_mask]  # [N_valid, 3]
        else:
            visible_mask = valid_mask
            points_3d = points_3d[valid_mask]  # [N_valid, 3]
            normals = normals[valid_mask]  # [N_valid, 3]
            area = area[valid_mask]  # [N_valid]
            velocity = velocity[valid_mask]  # [N_valid, 3]
        
        # Expand tx_position and rx_positions for batch processing
        tx_pos = self.tx_position.unsqueeze(0)  # [1, 3]
        rx_pos = self.rx_positions.unsqueeze(0)  # [1, num_rx, 3]
        
        k_i = (points_3d.unsqueeze(1) - tx_pos.unsqueeze(1))  # [N_valid, 1, 3]
        k_s = (rx_pos.unsqueeze(1) - points_3d.unsqueeze(2))  # [N_valid, num_rx, 3]
        
        k_i_length = torch.norm(k_i, dim=-1)  # [N_valid, 1]
        k_s_length = torch.norm(k_s, dim=-1)  # [N_valid, num_rx]
        distances = k_i_length + k_s_length  # [N_valid, num_rx]
        tau = distances / 2.99792458e8  # [N_valid, num_rx]
        
        if not consider_energy:  
            a = torch.ones_like(tau) 
        else: 
            # compute area
            k_i = k_i / k_i_length.unsqueeze(-1)  # [N_valid, 1, 3]
            k_s = k_s / k_s_length.unsqueeze(-1)  # [N_valid, num_rx, 3]
            cos_theta = torch.sum(k_i.squeeze(1) * normals, dim=-1)  # [N_valid]
            area = area * torch.abs(cos_theta)  # [N_valid]
            
            # compute energy loss according to the distance
            a = torch.ones_like(tau)  # [N_valid, num_rx]
            a = a / distances
            a = a * area.unsqueeze(-1)
            
            if self.scat_pattern is not None: 
                normals = torch.repeat_interleave(normals, self.num_rx, dim=0)  # [N_valid*num_rx, 3]
                k_i = k_i.expand(-1, self.num_rx, -1).reshape(-1, 3)  # [N_valid*num_rx, 3]
                k_s = k_s.reshape(-1, 3)  # [N_valid*num_rx, 3]
                coff_scat = self.scat_pattern(k_i=k_i, k_s=k_s, n_hat=normals)
                coff_scat = coff_scat.reshape(-1, self.num_rx)  # [N_valid, num_rx]
                a = a * coff_scat
                
            if self.radar_pattern is not None: 
                theta = torch.atan2(k_i[..., 1], k_i[..., 0])  # [N_valid, num_rx]
                phi = torch.atan2(k_i[..., 2], torch.norm(k_i[..., :2], dim=-1))  # [N_valid, num_rx]
                c_theta, c_phi = self.radar_pattern(
                    theta=np.pi/2 + theta.reshape(-1, 1),
                    phi=phi.reshape(-1, 1)
                )
                coff_radar = torch.abs(c_theta)**2 + torch.abs(c_phi)**2
                coff_radar = coff_radar.reshape(-1, self.num_rx)  # [N_valid, num_rx]
                a = a * coff_radar
        
        # Compute Doppler velocity
        vel = (torch.sum(velocity.unsqueeze(1) * k_i, dim=-1) - 
               torch.sum(velocity.unsqueeze(1) * k_s, dim=-1)) / 2  # [N_valid, num_rx]
        
        # Reshape outputs back to batch dimension
        num_points_per_batch = torch.sum(visible_mask, dim=1)  # [B]
        max_points = num_points_per_batch.max()
        
        # Initialize output tensors
        a_out = torch.zeros(batch_size, self.num_rx, max_points, device=depth_map.device)
        tau_out = torch.zeros(batch_size, self.num_rx, max_points, device=depth_map.device)
        vel_out = torch.zeros(batch_size, self.num_rx, max_points, device=depth_map.device)
        
        # Fill output tensors
        start_idx = 0
        for b in range(batch_size):
            n_points = num_points_per_batch[b]
            if n_points > 0:
                end_idx = start_idx + n_points
                a_out[b, :, :n_points] = a[start_idx:end_idx].T
                tau_out[b, :, :n_points] = tau[start_idx:end_idx].T
                vel_out[b, :, :n_points] = vel[start_idx:end_idx].T
                start_idx = end_idx
        
        return {
            'a': a_out.to(self.dtype), 
            'tau': tau_out.to(self.dtype), 
            'vel': vel_out.to(self.dtype)
        }


def get_index():
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
    TX = np.concatenate([[i] * 16 for i in range(12)])
    return D_uni, index, TX[index]


def get_full_index(): 
    D = [[i, j] for i in range(32) for j in range(32)]
    return D, None, None
