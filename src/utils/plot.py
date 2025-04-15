import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_hand(ax: plt.Axes, joints: torch.Tensor, title=None, show_index=True, plot_3d=False, s=0.1):
    
    if isinstance(joints, torch.Tensor): 
        joints = joints.detach().cpu().numpy()
    
    skeleton = [
        (0, 5), (5, 9), (9, 13), (13, 17), (17, 0), # gray
        (0, 1), (1, 2), (2, 3), (3, 4), # blue
        (5, 6), (6, 7), (7, 8), # red
        (9, 10), (10, 11), (11, 12), # yellow
        (13, 14), (14, 15), (15, 16), # green
        (17, 18), (18, 19), (19, 20), # purple
    ]
    tic = time.time()
    for k, (i, j) in enumerate(skeleton):
        color = 'black'
        if plot_3d: 
            ax.plot([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]], [joints[i, 2], joints[j, 2]], color=color)
        else: 
            ax.plot([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]], color=color)
    print('plot1', time.time() - tic); tic = time.time()
    
    for i in range(joints.shape[0]):
        if plot_3d: 
            ax.scatter(joints[i, 0], joints[i, 1], joints[i, 2], s=s, c='black', zorder=3)
        else: 
            ax.scatter(joints[i, 0], joints[i, 1], s=s, c='black', zorder=3)
            if show_index:  
                ax.text(joints[i, 0], joints[i, 1], str(i), fontsize=12, ha='right', zorder=4)
    print('plot2', time.time() - tic); tic = time.time()
    
    if title is not None: 
        ax.set_title(title)
    return ax


def plot_hand_camera(image, joints, camera_position, camera_target, camera_intrinsic, boundary=False, draw_skeleton=True):
    fx, fy, cx, cy = camera_intrinsic 
    
    # from points to default
    joint_to_default = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    
    forward = np.array(camera_target) - np.array(camera_position)
    forward = forward.astype(np.float32)
    forward /= np.linalg.norm(forward)
    roll = -np.arcsin(forward[2])
    pitch = -np.arctan2(forward[0], forward[1])
    yaw = 0
    roll_R = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    pitch_R = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    default_to_camera = roll_R @ pitch_R
    
    joints_default = (joints - camera_position) @ joint_to_default # to default camera coordinate
    joints_camera = joints_default @ default_to_camera.T # to current camera coordinate
    joints_camera[..., 0] = joints_camera[..., 0] / joints_camera[..., 2] * fx + cx
    joints_camera[..., 1] = joints_camera[..., 1] / joints_camera[..., 2] * fy + cy
    # save for depth
    joints_depth = 1 - (joints_default[..., 2] - joints_default[:, 2].min()) / (joints_default[:, 2].max() - joints_default[:, 2].min())

    skeleton = [
        (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20)
    ]
    
    items_to_draw = []
    
    if draw_skeleton:
        for i, j in skeleton:
            pt1 = joints_camera[i]
            pt2 = joints_camera[j]
            items_to_draw.append([
                (pt1[:2].astype(np.int32), pt2[:2].astype(np.int32)), 
                'line', 
                max(pt1[2], pt2[2])
            ])
    
    for i in range(len(joints_camera)):
        pt = joints_camera[i]
        items_to_draw.append([
            (pt[:2].astype(np.int32), 5), 
            'point', 
            joints_depth[i]
        ])
        
    if boundary: 
        boundary = np.concatenate([joints_default.min(0), joints_default.max(0)], 0)
        boundary_line = get_boundary(boundary)
        boundary_line = boundary_line @ default_to_camera.T
        boundary_line[..., 0] = boundary_line[..., 0] / boundary_line[..., 2] * fx + cx
        boundary_line[..., 1] = boundary_line[..., 1] / boundary_line[..., 2] * fy + cy
        for line in boundary_line: 
            pt1 = line[0]
            pt2 = line[1]
            items_to_draw.append([
                (pt1[:2].astype(np.int32), pt2[:2].astype(np.int32)), 
                'boundary', 
                max(pt1[2], pt2[2])
            ])
    
    # sort by z
    items_to_draw.sort(key=lambda x: x[2])
    for item in items_to_draw: 
        if item[1] == 'line': 
            cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), (0, 0, 0), 2)
        elif item[1] == 'point': 
            color = (0, 0, int(255 * item[2]))
            cv2.circle(image, tuple(item[0][0]), item[0][1], color, -1)
        elif item[1] == 'boundary':
            cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), (0, 0, 0), 1)
    return image

def plot_points(image, points, camera_position, camera_target, camera_intrinsic, color=(0, 0, 255), radius=5):
    """
    Project 3D points onto a 2D image plane using camera parameters.
    
    Args:
        image (np.ndarray): The image on which to plot the points.
        points (np.ndarray): The 3D points to be plotted.
        camera_position (list or np.ndarray): The position of the camera in 3D space.
        camera_target (list or np.ndarray): The target point the camera is looking at.
        camera_intrinsic (tuple): Intrinsic camera parameters (fx, fy, cx, cy).
        color (tuple or np.ndarray): The color of the points to be plotted. Can be a single color tuple, an array of colors, or grayscale values.
        radius (int): The radius of the points to be plotted.
    
    Returns:
        np.ndarray: The image with the plotted points.
    """
    fx, fy, cx, cy = camera_intrinsic
    
    # Define transformation matrices
    joint_to_default = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    
    # Calculate forward vector and rotation matrices
    forward = np.array(camera_target) - np.array(camera_position)
    forward = forward.astype(np.float32)
    forward /= np.linalg.norm(forward)
    roll = -np.arcsin(forward[2])
    pitch = -np.arctan2(forward[0], forward[1])
    roll_R = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    pitch_R = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    default_to_camera = roll_R @ pitch_R
    
    # Transform points to the default camera coordinate system
    points_default = (points - camera_position) @ joint_to_default
    points_camera = points_default @ default_to_camera.T
    points_camera[..., 0] = points_camera[..., 0] / points_camera[..., 2] * fx + cx
    points_camera[..., 1] = points_camera[..., 1] / points_camera[..., 2] * fy + cy
    
    # Determine colors for each point
    if isinstance(color, tuple):
        colors = [color] * len(points_camera)
    elif color.ndim == 1 and color.size > 1:
        # Grayscale values, map to colors using a colormap
        norm = plt.Normalize(vmin=color.min(), vmax=color.max())
        cmap = plt.get_cmap('viridis')
        colors = [cmap(norm(c))[:3] for c in color]  # Get RGB from colormap
        colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]  # Convert to BGR
    else:
        colors = color
    
    for pt, col in zip(points_camera, colors):
        cv2.circle(image, tuple(pt[:2].astype(np.int32)), radius, col, -1)
    
    return image


def get_boundary(boundary):
    x0, y0, z0, x1, y1, z1 = boundary + np.array([-10, -10, -10, 10, 10, 10])
    boundary_line = np.array([
        [x0, y0, z0], [x0, y0, z1], 
        [x0, y1, z0], [x0, y1, z1], 
        [x1, y0, z0], [x1, y0, z1], 
        [x1, y1, z0], [x1, y1, z1], 
        [x0, y0, z0], [x1, y0, z0], 
        [x0, y1, z0], [x1, y1, z0], 
        [x0, y0, z1], [x1, y0, z1], 
        [x0, y1, z1], [x1, y1, z1], 
        [x0, y0, z0], [x0, y1, z0], 
        [x1, y0, z0], [x1, y1, z0], 
        [x0, y0, z1], [x0, y1, z1], 
        [x1, y0, z1], [x1, y1, z1]
    ]).reshape(-1, 2, 3)
    return boundary_line


def plot_hand_subplot(position, camera_params, joints_l, joints_r, title):
    plt.subplot(position)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    img = plot_hand_camera(img, joints_l, **camera_params, boundary=True)
    img = plot_hand_camera(img, joints_r, **camera_params, boundary=True)
    plt.imshow(img)
    plt.title(title)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.xticks([]); plt.yticks([])


def show_result(joints, joints_pred, seq, idx):
    assert joints.shape == (2, 21, 3), "joints shape should be (2, 21, 3), but got %s" % str(joints.shape)
    joints_points_l = joints[0].detach().cpu().numpy()
    joints_points_r = joints[1].detach().cpu().numpy()
    joints_pred_points_l = joints_pred[0].detach().cpu().numpy()
    joints_pred_points_r = joints_pred[1].detach().cpu().numpy()
    
    # transform to default camera coordinate
    joints_points_l = joints_points_l @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    joints_points_r = joints_points_r @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    joints_pred_points_l = joints_pred_points_l @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    joints_pred_points_r = joints_pred_points_r @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            
    camera_params = {
        'camera_position': [100, -100, 100],
        'camera_target': [0, 300, 0],
        'camera_intrinsic': [454, 454, 256.0, 256.0]
    }
    plt.figure(dpi=200)
    plot_hand_subplot(121, camera_params, joints_points_l, joints_points_r, 'Ground Truth')
    plot_hand_subplot(122, camera_params, joints_pred_points_l, joints_pred_points_r, 'Prediction')
    os.makedirs('output/%s' % seq, exist_ok=True)
    plt.savefig('output/%s/%06d.png' % (seq, idx))
    plt.close()
    

def get_skeleton_length(joints):
    skeleton = [
        (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20)
    ]
    return np.array([np.linalg.norm(joints[i] - joints[j]) for i, j in skeleton])
