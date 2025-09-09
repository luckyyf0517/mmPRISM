import os
import time
import torch
import torch.nn as nn
import numpy as np

import sys; sys.path.append('.')

import matplotlib.pyplot as plt
from config.radar import iwr1843 as radar_cfg
from src.fmcw.beamformer import build_steering_vector, build_steering_vector_1d

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Processor(nn.Module):
    """Signal processing module for radar data"""
    def __init__(self, learnable_weights=False, W=32, H=32, dtype=torch.float32, ctype=torch.complex64, array_size="full"):
        super().__init__()
        self.array_size = array_size
        
        # Get antenna configuration based on array size
        if array_size == "small":
            D, self.antenna_indices = get_index_small()
        elif array_size == "middle":
            D, self.antenna_indices = get_index_middle()
        elif array_size == "large":
            D, self.antenna_indices = get_index_large()
        else:  
            D, self.antenna_indices = get_index_full()
            
        self.D_antennas = len(D)
        self.D, self.R, self.W, self.H = 64, 32, W, H
        
        # Initialize beamforming weights
        azi_ele_id = torch.tensor(np.array(D) - np.array([43, 0]))
        azi_theta_grid = torch.linspace(-np.pi/6, np.pi/6, self.W)
        ele_theta_grid = torch.linspace(-np.pi/6, np.pi/6, self.H)
        bm_weights = build_steering_vector(azi_ele_id, azi_theta_grid, ele_theta_grid)
        self.bm_weights = nn.Parameter(torch.view_as_real(bm_weights), requires_grad=learnable_weights)
        
        self.dtype = dtype
        self.ctype = ctype

        self.if_process_range = False
        self.if_process_doppler = True

    def process_range(self, radar_frame):
        """Range FFT processing
        Args:
            radar_frame: [B, num_chirps, num_antenna, num_samples]
        Returns:
            radar_frame: [B, num_chirps, num_antenna, R]
        """
        B = radar_frame.shape[0]
        num_samples = radar_frame.shape[-1]
        window = torch.hann_window(num_samples, device=radar_frame.device)
        window = window.view(1, 1, 1, -1).expand(B, -1, -1, -1)
        
        radar_frame = radar_frame * window
        radar_frame = torch.fft.fft(radar_frame, dim=-1)
        return radar_frame[..., :self.R]

    def process_doppler(self, radar_frame):
        """Doppler FFT processing
        Args:
            radar_frame: [B, num_chirps, num_antenna, R]
        Returns:
            radar_frame: [B, D, num_antenna, R]
        """
        B = radar_frame.shape[0]
        num_chirps = radar_frame.shape[1]
        
        # Remove DC component by subtracting mean across chirps
        radar_frame = radar_frame - radar_frame.mean(dim=1, keepdim=True)
        
        # Expand window to match batch dimension
        window = torch.hann_window(num_chirps, device=radar_frame.device)
        window = window.view(1, -1, 1, 1).expand(B, -1, -1, -1)
        
        radar_frame = radar_frame * window
        radar_frame = torch.fft.fftshift(torch.fft.fft(radar_frame, dim=1), dim=1)
        return radar_frame

    def process_beamforming(self, radar_frame):
        """Beamforming processing
        Args:
            radar_frame: [B, D, num_antenna, R]
        Returns:
            radar_frame: [B, D, W*H, R]
        """
        bm_weights = self.bm_weights[..., 0] + 1j * self.bm_weights[..., 1]  # [num_antenna, W*H]
        # Use einsum to process batch data
        radar_frame = torch.einsum('bdar,aw->bdrw', radar_frame, bm_weights)
        return radar_frame.abs() ** 2

    def forward(self, raw_radar_frame):
        """Convert radar frame to mmwave cube
        
        Args:
            raw_radar_frame: [B, num_chirps, num_antenna, num_samples] Complex radar signals
            
        Returns:
            radar_frame: [B, D, R, W, H] Processed radar signals
        """
        # Select appropriate antennas based on array size configuration
        if hasattr(self, 'antenna_indices') and len(self.antenna_indices) < raw_radar_frame.shape[2]:
            # Use the precomputed indices to select antennas
            radar_frame = raw_radar_frame[:, :, self.antenna_indices, :].clone()
        else:
            # Use all antennas (default behavior)
            radar_frame = raw_radar_frame.clone()
        
        # Process step by step
        if self.if_process_range:
            radar_frame = self.process_range(radar_frame)    # [B, num_chirps, num_antenna, R]
        if self.if_process_doppler:
            radar_frame = self.process_doppler(radar_frame)  # [B, D, num_antenna, R]
        radar_frame = self.process_beamforming(radar_frame)  # [B, D, W*H, R]
        
        # Reshape output dimensions
        B = radar_frame.shape[0]
        return radar_frame.view(B, self.D, self.R, self.W, self.H)


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
        
        self.dtype = dtype
        self.ctype = ctype

    def simulate_batch(self, points_3d, velocities_3d):
        """Process single batch data"""

        path_dict = self.simulator.compute_paths_from_points(
            points_3d=points_3d,
            velocities_3d=velocities_3d)
        
        return self.get_raw_radar_frame(path_dict)

    def forward(self, points_3d, velocities_3d):
        """
        Forward pass using 3D points and velocities
        
        Args:
            points_3d: [B, N, 3] 3D points in camera coordinates for batch
            velocities_3d: [B, N, 3] 3D velocities in m/s for batch
            
        Returns:
            raw_radar_frames: [B, num_chirps, num_rx, num_samples] raw radar signals
        """
        batch_size = points_3d.shape[0]
        radar_frame_list = []
        
        processed_velocities, nan_mask = process_point_cloud(velocities_3d)
        processed_points, _ = process_point_cloud(points_3d, nan_mask)
        assert torch.isnan(processed_points).any() == False, 'processed_points has NaN values'
        assert torch.isnan(processed_velocities).any() == False, 'processed_velocities has NaN values'

        for b in range(batch_size):
            try:
                radar_frame = self.simulate_batch(processed_points[b], processed_velocities[b])
                radar_frame_list.append(radar_frame)
            except:
                print(f"Error processing batch {b}")
                raise
            
        return torch.stack(radar_frame_list, dim=0)
    
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
        # tau_chirp: [num_chirps, num_rx, 1, max_num_paths]
        tau_chirp = tau_velocity * time_steps.to(a.device)  
        tau_chirp = tau.unsqueeze(0) + tau_chirp
        tau_chirp = tau_chirp[:, :, None, :] 
        # frequencies: [1, 1, num_samples, 1]
        frequencies = self.fs[None, None, :, None].to(a.device)
        # ft_phase: [num_chirps, num_rx, num_samples, max_num_paths]
        ft_phase = 2 * np.pi * (frequencies + self.start_freq) * tau_chirp
        ft_phase %= (2 * np.pi)
        # a: [1, num_rx, 1, max_num_paths]
        a = a[None, :, None, :]
        # radar_frame: [num_chirps, num_rx, num_samples]
        if save_cuda_memory: 
            a = a.to(torch.float16)
            ft_phase = ft_phase.to(torch.float16)
        radar_frame = (a * torch.exp(1j * ft_phase)).sum(dim=-1)
        return radar_frame.to(self.dtype)
    

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
        # Set the radar position
        self.radar_position = np.array([0.0, 0.0, -0.80])  
        # Set the camera position
        self.camera_position = self.radar_position + np.array([0.0, 0.0, 0.0])
        
        # Get the index of the radar
        self.D = np.array(get_index()) - np.array([43, 0])
        # Calculate the wavelength
        d = (2.99792458e8 / self.frequency) / 2     
        # Calculate the reference array
        rx_ref_array = np.column_stack((self.D[:, 0], self.D[:, 1], np.zeros(len(self.D))))
        rx_ref_array = rx_ref_array * d
        
        # Register the tx position
        self.register_buffer('tx_position', torch.tensor(self.radar_position), persistent=False)
        # Register the rx positions
        self.register_buffer('rx_positions', torch.tensor(self.radar_position[None] + rx_ref_array), persistent=False)
        # Initialize the scatter pattern and radar pattern
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
        # Calculate the incident wave vector
        k_i = points_3d - self.tx_position  # [N, 3]
        # Calculate the scattered wave vector
        k_s = (self.rx_positions[:, None] - points_3d)  # [num_rx, N, 3]
        
        # Calculate the distance
        k_i_length = torch.norm(k_i, dim=-1)  # [N]
        k_s_length = torch.norm(k_s, dim=-1)  # [num_rx, N]
        distances = k_i_length[None] + k_s_length  # [num_rx, N]
        
        # Calculate the time delay
        tau = distances / 2.99792458e8  # [num_rx, N]
        # Calculate the amplitude
        a = torch.ones_like(tau) / (distances / 2) ** 2
        
        # Calculate the radial velocity
        k_i = k_i / k_i_length.unsqueeze(-1)  # [N, 3]
        k_s = k_s / k_s_length.unsqueeze(-1)  # [num_rx, N, 3]
        vel = (torch.sum(velocities_3d * k_i, dim=-1)[None] - 
               torch.sum(velocities_3d * k_s, dim=-1)) / 2  # [num_rx, N]
        
        return {
            'a': a.to(self.dtype),
            'tau': tau.to(self.dtype),
            'vel': vel.to(self.dtype)
        }


def get_index_full():

    D = [
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0], 
        [11, 1], [11, 4], [11, 6], [12, 0], [12, 1], [12, 4], [12, 6], [13, 0], [14, 0], [15, 0], [16, 0], 
        [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0], [22, 1], [22, 4], [22, 6], [23, 0], [23, 1], 
        [23, 4], [23, 6], [24, 0], [25, 0], [26, 0], [27, 0], [28, 0], [29, 0], [30, 0], [31, 0], [32, 0], 
        [33, 0], [34, 0], [35, 0], [36, 0], [37, 0], [38, 0], [39, 0], [40, 0], [41, 0], [42, 0], [43, 0], 
        [44, 0], [45, 0], [46, 0], [47, 0], [48, 0], [49, 0], [50, 0], [51, 0], [52, 0], [53, 0], [54, 0], 
        [55, 0], [56, 0], [57, 0], [57, 1], [57, 4], [57, 6], [58, 0], [58, 1], [58, 4], [58, 6], [59, 0], 
        [59, 1], [59, 4], [59, 6], [60, 0], [60, 1], [60, 4], [60, 6], [61, 0], [61, 1], [61, 4], [61, 6], 
        [62, 0], [62, 1], [62, 4], [62, 6], [63, 0], [64, 0], [65, 0], [66, 0], [67, 0], [68, 0], [69, 0], 
        [70, 0], [71, 0], [72, 0], [73, 0], [74, 0], [75, 0], [76, 0], [77, 0], [78, 0], [79, 0], [80, 0], 
        [81, 0], [82, 0], [83, 0], [84, 0], [85, 0]]

    return D, np.arange(0, len(D))


def get_index_large():

    D = [
        [11, 0], 
        [11, 1], [11, 4], [11, 6], [12, 0], [12, 1], [12, 4], [12, 6], [13, 0], [14, 0], [15, 0], [16, 0], 
        [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0], [22, 1], [22, 4], [22, 6], [23, 0], [23, 1], 
        [23, 4], [23, 6], [24, 0], [25, 0], [26, 0], [27, 0], [28, 0], [29, 0], [30, 0], [31, 0], [32, 0], 
        [33, 0], [34, 0], [35, 0], [36, 0], [37, 0], [38, 0], [39, 0], [40, 0], [41, 0], [42, 0]]

    D_full, _ = get_index_full()
    index = [D_full.index(item) for item in D]

    return D, index


def get_index_middle():

    D = [
        [11, 0], 
        [11, 1], [11, 4], [12, 0], [12, 1], [12, 4], [13, 0], [14, 0], [15, 0], [16, 0], 
        [17, 0], [18, 0], [19, 0], [20, 0], [21, 0], [22, 0], [22, 1], [22, 4], [23, 0], [23, 1], 
        [23, 4], [24, 0], [25, 0], [26, 0]]

    D_full, _ = get_index_full()
    index = [D_full.index(item) for item in D]

    return D, index


def get_index_small():

    D = [
        [11, 0], 
        [11, 1], [12, 0], [12, 1], [13, 0], [14, 0], [15, 0], [16, 0], 
        [17, 0], [18, 0]]

    D_full, _ = get_index_full()
    index = [D_full.index(item) for item in D]

    return D, index


def process_point_cloud(data, nan_mask=None):
    """
    Process a point cloud data of shape [T, 2, 24, 3] and return a new point cloud.
    
    Args:
        data: torch.Tensor of shape [T, 2, 24, 3] representing the point cloud data.
              Each side has 3 body points (arm) and 21 hand points.
        nan_mask: Optional pre-computed NaN mask to apply. If None, a new mask is created.
        
    Returns:
        torch.Tensor representing the processed point cloud, and the NaN mask used.
    """
    # Extract body and hand points for both sides
    left_body = data[:, 0, :3, :]   # Left arm points
    left_hand = data[:, 0, 3:, :]   # Left hand points
    right_body = data[:, 1, :3, :]  # Right arm points
    right_hand = data[:, 1, 3:, :]  # Right hand points

    # Define body skeleton for interpolation
    # Connections within left arm
    left_skeleton = torch.tensor([
        [0, 1], [1, 2]
    ], device=data.device)
    
    # Connections within right arm
    right_skeleton = torch.tensor([
        [0, 1], [1, 2]
    ], device=data.device)
    
    # Connection between left and right sides
    cross_skeleton = torch.tensor([
        [0, 0]  # Connect first point of left arm to first point of right arm
    ], device=data.device)

    # Vectorized interpolation of points
    def interpolate_points_vectorized(p1, p2, num_points=3):
        t_values = torch.linspace(0, 1, num_points + 2, device=data.device)[1:-1]
        t_values = t_values.unsqueeze(0).unsqueeze(-1)  # Reshape t_values for broadcasting
        return p1.unsqueeze(1) + (p2 - p1).unsqueeze(1) * t_values

    # Interpolate points for left arm
    left_interpolated = []
    for i, j in left_skeleton:
        interpolated_points = interpolate_points_vectorized(left_body[:, i], left_body[:, j])
        left_interpolated.append(interpolated_points)
    
    # Interpolate points for right arm
    right_interpolated = []
    for i, j in right_skeleton:
        interpolated_points = interpolate_points_vectorized(right_body[:, i], right_body[:, j])
        right_interpolated.append(interpolated_points)
    
    # Interpolate points between left and right arms
    cross_interpolated = []
    for i, j in cross_skeleton:
        interpolated_points = interpolate_points_vectorized(left_body[:, i], right_body[:, j])
        cross_interpolated.append(interpolated_points)
    
    # Combine all interpolated points
    interpolated_points = []
    if left_interpolated:
        interpolated_points.append(torch.cat(left_interpolated, dim=1))
    if right_interpolated:
        interpolated_points.append(torch.cat(right_interpolated, dim=1))
    if cross_interpolated:
        interpolated_points.append(torch.cat(cross_interpolated, dim=1))
    
    interpolated_points = torch.cat(interpolated_points, dim=1)
    
    # Combine all points: body points, interpolated points, and hand points
    all_points = torch.cat([
        left_body, right_body, 
        interpolated_points,
        left_hand, right_hand
    ], dim=1)
    
    # Create mask for points with NaN values if not provided
    if nan_mask is None:
        nan_mask = ~torch.isnan(all_points).any(dim=-1)  # [T, N]
    
    # Vectorized filtering of NaN values
    batch_size = all_points.shape[0]
    
    # If all points are filtered out, return a minimal valid tensor to avoid errors
    if not nan_mask.any():
        return torch.zeros((batch_size, 1, 3), device=data.device), nan_mask
    
    # Find the maximum number of valid points across batches
    num_valid_per_batch = nan_mask.sum(dim=1)  # [T]
    max_valid_points = num_valid_per_batch.max().item()
    
    if max_valid_points == 0:
        return torch.zeros((batch_size, 1, 3), device=data.device), nan_mask
    
    # Create output tensor
    padded_points = torch.zeros((batch_size, max_valid_points, 3), device=data.device)
    
    # Use a vectorized approach to filter and reshape
    for t in range(batch_size):
        # Get valid points for this batch item
        valid_count = num_valid_per_batch[t].item()
        if valid_count > 0:
            # Extract valid points directly using the mask
            padded_points[t, :valid_count] = all_points[t, nan_mask[t]]
    
    return padded_points, nan_mask
