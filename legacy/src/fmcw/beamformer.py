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
    
def build_steering_vector(azi_ele_id, azi_theta_grid, ele_theta_grid, device='cuda'):
    azi_ele_id = azi_ele_id.unsqueeze(1)
    azi_ele_theta = torch.cartesian_prod(azi_theta_grid, ele_theta_grid).unsqueeze(0)
    steering_vector = torch.exp(-1j * _phi_2d(
        azi_ele_theta[:, :, 0], azi_ele_theta[:, :, 1], azi_ele_id[:, :, 0], azi_ele_id[:, :, 1]))
    return steering_vector

def build_steering_vector_1d(azi_id, azi_theta_grid, device='cuda'):
    azi_theta = azi_theta_grid.unsqueeze(0)
    azi_id = azi_id.unsqueeze(1)
    steering_vector = torch.exp(-1j * _phi_2d(azi_theta, azi_theta, azi_id, azi_id))
    return steering_vector