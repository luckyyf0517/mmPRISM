import os
import sys; sys.path.append('/root/autodl-tmp/mmHand')

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from src.fmcw.simulator import get_index

f = 77e9
c = 2.99792458e8
lb = c / f
d = lb / 2


def _phi_2d(azi_theta, ele_theta, azi_idx, ele_idx):
    """When r is large, phi approximates to np.pi * k * sin(theta)"""
    phi = np.pi * azi_idx * torch.sin(azi_theta) + \
          np.pi * ele_idx * torch.cos(azi_theta) * torch.sin(ele_theta)
    return phi
    
    
def _phi_3d(r, azi_theta, ele_theta, azi_idx, ele_idx):
    """Calculate 3D phase difference (using Fresnel approximation)"""
    item = 2 * d * azi_idx * torch.sin(azi_theta) / r + \
           2 * d * ele_idx * torch.cos(azi_theta) * torch.sin(ele_theta) / r + \
           d**2 * (azi_idx**2 + ele_idx**2) / r**2 + 1
    phi = 2 * np.pi / lb * r * (torch.sqrt(item) - 1)
    return phi


def build_steering_vector_batch(azi_theta_grid, ele_theta_grid, device='cuda'):
    batch_size = azi_theta_grid.shape[0]
    azi_ele_id = torch.tensor(get_index()[0]) - torch.tensor([43, 0])
    azi_ele_id = azi_ele_id.to(device)

    azi_theta = azi_theta_grid.unsqueeze(2).expand(-1, -1, ele_theta_grid.shape[1]).reshape(batch_size, -1)
    ele_theta = ele_theta_grid.unsqueeze(1).expand(-1, azi_theta_grid.shape[1], -1).reshape(batch_size, -1)

    azi_idx = azi_ele_id[:, 0].unsqueeze(0).unsqueeze(-1)  # (1, num_elements, 1)
    ele_idx = azi_ele_id[:, 1].unsqueeze(0).unsqueeze(-1)  # (1, num_elements, 1)
    steering_vector = torch.exp(-1j * _phi_2d(azi_theta.unsqueeze(1), ele_theta.unsqueeze(1), azi_idx, ele_idx))
    return steering_vector


class BaseBeamformer(torch.nn.Module):
    def __init__(self, azi_theta_grid, ele_theta_grid, device='cuda'):
        super().__init__()
        self.azi_ele_id = torch.tensor(get_index()[0]) - torch.tensor([43, 0])
        self.azi_theta_grid = azi_theta_grid
        self.ele_theta_grid = ele_theta_grid
        self.device = device
        self.steering_vector = self._compute_steering_vector()
        
    
    def _compute_steering_vector(self):
        """Compute steering vector, implemented by subclasses"""
        raise NotImplementedError
    
    def beamforming_weights(self, X):
        """Compute beamforming weights, implemented by subclasses"""
        raise NotImplementedError
    
    def beamforming_spectrum(self, X):
        """Compute beamforming spectrum, implemented by subclasses"""
        raise NotImplementedError


class BartlettBeamformer2D(BaseBeamformer):
    def __init__(self, azi_theta_grid, ele_theta_grid, device='cuda'):
        super().__init__(azi_theta_grid, ele_theta_grid, device)
    
    def _compute_steering_vector(self, r_refine=None):
        """Compute steering vector for 2D Bartlett beamformer"""
        azi_ele_theta_grid = torch.cartesian_prod(self.azi_theta_grid, self.ele_theta_grid)
        steering_vector = torch.zeros(
            size=(len(self.azi_ele_id), len(azi_ele_theta_grid)), 
            dtype=torch.complex64, device=self.device)
        for k in range(len(self.azi_ele_id)):
            # For FMCW radar, the steering vector is (j 2 pi d sin(theta) / c)
            azi_idx, ele_idx = self.azi_ele_id[k]
            azi_theta, ele_theta = azi_ele_theta_grid[:, 0], azi_ele_theta_grid[:, 1]
            steering_vector[k] = torch.exp(-1j * _phi_2d(azi_theta, ele_theta, azi_idx, ele_idx))
            if r_refine is not None:
                steering_vector[k] = torch.exp(-1j * _phi_3d(r_refine, azi_theta, ele_theta, azi_idx, ele_idx))
        return steering_vector
        
    def beamforming_weights(self, X):
        return self.steering_vector
    
    def beamforming_spectrum(self, X):
        R = torch.matmul(X, X.conj().T)
        R = R / X.shape[1]
        bm_weights = self.beamforming_weights(X).to(X.dtype)
        spatial_spectrum = bm_weights.conj().T @ R @ bm_weights
        spatial_spectrum = torch.diag(spatial_spectrum).abs()
        spatial_spectrum = spatial_spectrum.reshape(len(self.azi_theta_grid), len(self.ele_theta_grid))
        return spatial_spectrum


class BartlettBeamformerNearfield(BaseBeamformer):
    def __init__(self, azi_theta_grid, ele_theta_grid, device='cuda'):
        self.r_grid = torch.linspace(0.2, 0.4, 8) * 8
        super().__init__(azi_theta_grid, ele_theta_grid, device)
    
    def _compute_steering_vector(self): 
        """Compute steering vector for near-field Bartlett beamformer"""
        azi_ele_r_theta_grid = torch.cartesian_prod(self.azi_theta_grid, self.ele_theta_grid, self.r_grid)
        steering_vector = torch.zeros(
            size=(len(self.azi_ele_id), len(azi_ele_r_theta_grid)), 
            dtype=torch.complex64, device=self.device)
        for k in range(len(self.azi_ele_id)):
            azi_idx, ele_idx = self.azi_ele_id[k]
            azi_theta, ele_theta, r = azi_ele_r_theta_grid[:, 0], azi_ele_r_theta_grid[:, 1], azi_ele_r_theta_grid[:, 2]
            steering_vector[k] = torch.exp(-1j * _phi_3d(r, azi_theta, ele_theta, azi_idx, ele_idx))
        return steering_vector
        
    def beamforming_weights(self, X):
        return self.steering_vector
        
    def beamforming_spectrum(self, X):
        R = torch.matmul(X, X.conj().T)
        R = R / X.shape[1]
        bm_weights = self.beamforming_weights(X)
        spatial_spectrum = bm_weights.conj().T @ R @ bm_weights
        spatial_spectrum = torch.diag(spatial_spectrum).real
        spatial_spectrum = spatial_spectrum.reshape(len(self.azi_theta_grid), len(self.ele_theta_grid), len(self.r_grid))
        return spatial_spectrum.max(-1).values


class CaponBeamformer2D(BaseBeamformer):
    def __init__(self, azi_theta_grid, ele_theta_grid, device='cuda'):
        super().__init__(azi_theta_grid, ele_theta_grid, device)
    
    def _compute_steering_vector(self):
        """Compute steering vector for 2D Capon beamformer"""
        azi_ele_theta_grid = torch.cartesian_prod(self.azi_theta_grid, self.ele_theta_grid)
        steering_vector = torch.zeros(
            size=(len(self.azi_ele_id), len(azi_ele_theta_grid)), 
            dtype=torch.complex64, device=self.device)
        for k in range(len(self.azi_ele_id)):
            azi_idx, ele_idx = self.azi_ele_id[k]
            azi_theta, ele_theta = azi_ele_theta_grid[:, 0], azi_ele_theta_grid[:, 1]
            steering_vector[k] = torch.exp(-1j * _phi_2d(azi_theta, ele_theta, azi_idx, ele_idx))
        return steering_vector
    
    def beamforming_weights(self, X):
        R = torch.matmul(X, X.conj().T)
        R = R / X.shape[1]
        R_inv = torch.linalg.pinv(R)
        # Beamforming
        num = R_inv @ self.steering_vector
        den = self.steering_vector.conj().T @ R_inv @ self.steering_vector
        bm_weights = num @ torch.linalg.pinv(den)
        return bm_weights
    
    def beamforming_spectrum(self, X):
        R = torch.matmul(X, X.conj().T)
        R = R / X.shape[1]
        bm_weights = self.beamforming_weights(X)
        spatial_spectrum = bm_weights.conj().T @ R @ bm_weights
        spatial_spectrum = torch.diag(spatial_spectrum).real
        spatial_spectrum = spatial_spectrum.reshape(len(self.azi_theta_grid), len(self.ele_theta_grid))
        return spatial_spectrum
