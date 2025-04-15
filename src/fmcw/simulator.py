import os
import time
import torch
import torch.nn as nn
import trimesh
import numpy as np

import matplotlib.pyplot as plt
import sys; sys.path.append('.')

from src.fmcw.fmcw_radar import FMCWRadar


class mmSimulator(nn.Module): 
    def __init__(self, radar_cfg, dtype=torch.float32, ctype=torch.complex64): 
        super().__init__()
        # Load integrated scene
        self.radar = FMCWRadar(radar_cfg, dtype=dtype, ctype=ctype)
        self.frequency = radar_cfg.start_freq
        # Equivalent to 3tx and 4rx
        self.num_tx = 1
        self.num_rx = 12
        # hand mano mesh 
        self.v_num = 778
        self.f_num = 1538
        
        self.render = False
        self.mesh = None
        self.area = None
        self.faces_l = None
        self.faces_r = None
        self.normals = None
        self.vertices = None
        self.sample_ratio = 1.0
        
        self.dtype = dtype
        self.ctype = ctype

    def init(self):
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
        
    def compute_paths_using_points(self, vertices_l, velocity_l, vertices_r=None, velocity_r=None, consider_energy=True, remove_unvisible=True): 
        """
        Compute path delay using distances. No physics considered.  
        a: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
        tau: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        """
        hit_points_l = self.generate_mesh_tracking_points(vertices_l[self.faces_l])
        area_l, normals_l = self.compute_mesh_triangle_area_and_normals(vertices_l[self.faces_l])
        if vertices_r is not None:  
            hit_points_r = self.generate_mesh_tracking_points(vertices_r[self.faces_r])
            area_r, normals_r = self.compute_mesh_triangle_area_and_normals(vertices_r[self.faces_r])
            hit_points = torch.cat([hit_points_l, hit_points_r], dim=0)
            area = torch.cat([area_l, area_r], dim=0)
            normals = torch.cat([normals_l, normals_r], dim=0)
            velocity = torch.cat([velocity_l[self.faces_l], velocity_r[self.faces_r]], dim=0)
        else: 
            hit_points = hit_points_l
            area = area_l
            normals = normals_l
            velocity = velocity_l[self.faces_l]
        
        if remove_unvisible: 
            view_direction = self.tx_position - hit_points
            dot_product = torch.sum(normals * view_direction, axis=-1)
            visible_mask = dot_product > 0
            hit_points = hit_points[visible_mask]
            normals = normals[visible_mask]
            area = area[visible_mask]
        
        k_i = (hit_points - self.tx_position)[None]
        k_s = (self.rx_positions[:, None] - hit_points)
        k_i_length = torch.norm(k_i, dim=-1)
        k_s_length = torch.norm(k_s, dim=-1)
        distances = k_i_length[:, None] + k_s_length[None]
        tau = distances / 2.99792458e8
        if not consider_energy:  
            a = torch.ones_like(tau) 
        else: 
            # compute area
            k_i = k_i / k_i_length[:, :, None]
            k_s = k_s / k_s_length[:, :, None]
            cos_theta = torch.sum(k_i[0] * normals, dim=-1) / k_i_length[0]
            area = area * torch.abs(cos_theta)
            # compute energy loss according to the distance
            a = torch.ones_like(tau) 
            a = a / distances
            a = a * area
            if self.scat_pattern is not None: 
                normals = torch.repeat_interleave(normals, 12, dim=0)
                k_i = torch.repeat_interleave(k_i[None], 12, dim=1)
                k_s = k_s[None]
                coff_scat = self.scat_pattern(k_i=torch.reshape(k_i, (-1, 3)), k_s=torch.reshape(k_s, (-1, 3)), n_hat=normals)
                coff_scat = torch.reshape(coff_scat, k_i.shape[:-1])
                a = a * coff_scat
            if self.radar_pattern is not None: 
                theta = torch.atan2(k_i[..., 1], k_i[..., 0])
                phi = torch.atan2(k_i[..., 2], torch.norm(k_i[..., :2], dim=-1))
                c_theta, c_phi = self.radar_pattern(theta=np.pi/2 + torch.reshape(theta, (-1, 1)), phi=torch.reshape(phi, (-1, 1)))
                coff_radar = torch.abs(c_theta)**2 + torch.abs(c_phi)**2
                coff_radar = torch.reshape(coff_radar, k_i.shape[:-1])
                a = a * coff_radar
        vel = self.generate_mesh_tracking_points(velocity)
        vel = vel[visible_mask]
        vel = (torch.sum(vel * k_i, dim=-1) - torch.sum(vel * k_s, dim=-1)) / 2
        return {'a': a.to(self.dtype), 'tau': tau.to(self.dtype), 'vel': vel.to(self.dtype)}
    
    def generate_mesh_tracking_points(self, triangle_vpositions): 
        triangle_cpositions = torch.mean(triangle_vpositions, dim=1)
        return triangle_cpositions
    
    def compute_mesh_triangle_area_and_normals(self, triangle_vpositions): 
        # calculate the area of each triangle
        v1 = triangle_vpositions[:, 1] - triangle_vpositions[:, 0]
        v2 = triangle_vpositions[:, 2] - triangle_vpositions[:, 0]
        cross = torch.linalg.cross(v1, v2)
        area = 0.5 * torch.norm(cross, dim=1)
        normals = cross / torch.norm(cross, dim=1, keepdim=True)
        return area, normals
        
    def update_mesh_faces_left(self, faces): 
        self.faces_l = torch.tensor(faces.astype(np.int64))
        self.faces_l = self.faces_l[::int(1/self.sample_ratio)]
        
    def update_mesh_faces_right(self, faces): 
        self.faces_r = torch.tensor(faces.astype(np.int64))
        self.faces_r = self.faces_r[::int(1/self.sample_ratio)]


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
