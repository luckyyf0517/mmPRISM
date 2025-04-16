import os
import sys; sys.path.append('/root/autodl-tmp/mmHand')

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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


def build_steering_vector(azi_ele_id, azi_theta_grid, ele_theta_grid, device='cuda'):
    azi_ele_id = azi_ele_id.unsqueeze(1)
    azi_ele_theta = torch.cartesian_prod(azi_theta_grid, ele_theta_grid).unsqueeze(0)
    steering_vector = torch.exp(-1j * _phi_2d(
        azi_ele_theta[:, :, 0], azi_ele_theta[:, :, 1], azi_ele_id[:, :, 0], azi_ele_id[:, :, 1]))
    return steering_vector
