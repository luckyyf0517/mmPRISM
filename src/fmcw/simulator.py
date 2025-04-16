import os
import time
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
from configs.radar import iwr1843 as radar_cfg
from src.fmcw.beamformer import build_steering_vector


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
        
        # self.H = 64
        # self.W = 64
        # self.steering_vector = build_steering_vector(
        #     azi_ele_id=torch.from_numpy(self.simulator.D),
        #     azi_theta_grid=torch.linspace(-np.pi/3, np.pi/3, self.H),
        #     ele_theta_grid=torch.linspace(-np.pi/4, np.pi/4, self.W),
        # ).to(torch.complex64)
        
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
        mmwave_cube = self.get_mmwave_cube(radar_frame)
        return mmwave_cube
    
    def get_raw_radar_frame(self, path_dict, save_cuda_memory=False):
        """
        Generate raw radar frame from path information
        
        Args:
            path_dict: Dictionary containing path information with keys:
                'a': [max_num_paths] amplitude 
                'tau': [max_num_paths] time delay
                'vel': [max_num_paths] velocity
            save_cuda_memory: Whether to use memory-efficient computation
            
        Returns:
            radar_frame: [num_chirps, num_samples] complex radar signals
        """
        a = path_dict['a']  # [max_num_paths]
        tau = path_dict['tau']  # [max_num_paths]
        vel = path_dict['vel']  # [max_num_paths]
        tau_velocity = vel * 2 / 2.99792458e8
        time_steps = self.time_steps[:, None]
            
        tau_chirp = tau + tau_velocity * time_steps.to(a.device)
        tau_chirp = tau_chirp.unsqueeze(-2)  # [num_chirps, 1, max_num_paths]
        frequencies = self.fs.unsqueeze(0).unsqueeze(-1).to(a.device)  # [1, num_samples, 1]
        ft_phase = 2 * np.pi * (frequencies + self.start_freq) * tau_chirp
        ft_phase %= (2 * np.pi)
        
        radar_frame = (a * torch.exp(1j * ft_phase)).sum(dim=-1)  # [num_chirps, num_samples]
        return radar_frame
    
    def get_mmwave_cube(self, raw_radar_frame):
        """
        Convert radar frame to mmwave cube
        """
        # num_chirps, num_antenna, num_samples = radar_frame.shape
        radar_frame = raw_radar_frame.clone()
        radar_frame = radar_frame * torch.hann_window(256, device=radar_frame.device)
        radar_frame = torch.fft.fft(radar_frame, dim=-1)
        radar_frame = radar_frame[:, :64]
        radar_frame = radar_frame * torch.hann_window(64, device=radar_frame.device)[:, None]
        radar_frame = torch.fft.fftshift(torch.fft.fft(radar_frame, dim=0), dim=0)
        radar_frame = radar_frame.abs() ** 2
        
        from IPython import embed; embed(header='get_mmwave_cube')
        plt.imshow(radar_frame.cpu().numpy().T, aspect='auto')
        plt.savefig(f'output.png')
        
        return radar_frame


class mmSimulator(nn.Module): 
    def __init__(self, radar_cfg, dtype=torch.float32, ctype=torch.complex64): 
        super().__init__()
        # Load integrated scene
        self.frequency = radar_cfg.start_freq
        # Single antenna
        self.num_tx = 1
        self.num_rx = 1
        
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
        
        self.radar_position = np.array([0.0, 0.0, 0.0])  
        self.camera_position = self.radar_position + np.array([0.0, 0.0, 0.0])
        
        # Single antenna position
        self.register_buffer('tx_position', torch.tensor(self.radar_position), persistent=False)
        self.register_buffer('rx_positions', torch.tensor(self.radar_position)[None], persistent=False)
        self.scat_pattern = None
        self.radar_pattern = None
        
    def depth_to_points(self, depth_map):
        """
        Convert depth map to 3D points in camera coordinates using grid computation
        
        Args:
            depth_map: [H, W] depth values in meters
            
        Returns:
            points: [N, 3] 3D points in camera coordinates
            valid_mask: [H*W] boolean mask for valid points
        """
        height, width = depth_map.shape
        
        # Create pixel coordinate grid [H, W]
        v, u = torch.meshgrid(
            torch.arange(height, device=depth_map.device),
            torch.arange(width, device=depth_map.device),
            indexing='ij'
        )
        
        # Calculate normalized camera coordinates [H, W]
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        
        # Process depth
        depths = depth_map - 1.20  # [H, W]
        
        # Calculate 3D point cloud [H, W, 3]
        points_3d = torch.stack([
            x * depths,      # X = (u-cx)/fx * Z
            y * depths,      # Y = (v-cy)/fy * Z
            depths           # Z
        ], dim=-1)
        
        # Reshape to [H*W, 3] and create valid points mask
        points_3d = points_3d.reshape(-1, 3)
        valid_mask = depths.reshape(-1) > 0
        
        return points_3d, valid_mask
        
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
        # Single antenna position
        tx_pos = self.tx_position  # [3]
        
        # Calculate incident and scattered vectors
        k_i = points_3d - tx_pos  # [N, 3]
        k_s = -k_i  # [N, 3] (backscattering)
        
        # Calculate distances and normalize vectors
        k_i_length = torch.norm(k_i, dim=-1)  # [N]
        k_i = k_i / k_i_length.unsqueeze(-1)  # [N, 3]
        distances = 2 * k_i_length  # [N] (round trip)
        tau = distances / 2.99792458e8  # [N]
        a = torch.ones_like(tau)
        
        # Compute Doppler velocity (radial component)
        vel = torch.sum(velocities_3d * k_i, dim=-1)  # [N]
        
        return {
            'a': a.to(self.dtype),
            'tau': tau.to(self.dtype),
            'vel': vel.to(self.dtype)
        }