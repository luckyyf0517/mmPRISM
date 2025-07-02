import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_hand_camera(image, joints, camera_position, camera_intrinsic, boundary=False, draw_skeleton=True, colors=None):
    fx, fy, cx, cy = camera_intrinsic 
    
    # Calculate positions relative to camera
    joints_camera = joints - camera_position
    
    # Project to image plane
    joints_image = joints_camera.copy()
    joints_image[..., 0] = joints_image[..., 0] / joints_image[..., 2] * fx + cx
    joints_image[..., 1] = joints_image[..., 1] / joints_image[..., 2] * fy + cy
    # Calculate depth values for shading
    valid_joints = ~np.isnan(joints_camera[:, 2])
    joints_depth = np.zeros_like(joints_camera[:, 2])
    if valid_joints.any():
        joints_depth[valid_joints] = 1 - (joints_camera[valid_joints, 2] - joints_camera[valid_joints, 2].min()) / (joints_camera[valid_joints, 2].max() - joints_camera[valid_joints, 2].min())

    # Normalize colors to 0-1 range if provided
    if colors is not None:
        colors = [c / max(colors) * 255 for c in colors]

    # Skeleton connection definitions
    skeleton = [
        # Arm skeleton (0,1,2)
        (0, 1), (1, 2),
        # Hand skeleton (starting from index 3, corresponding to original 0)
        (2, 3),  # Arm endpoint connects to palm
        (3, 8), (8, 12), (12, 16), (16, 20), (20, 3),  # Palm
        (3, 4), (4, 5), (5, 6), (6, 7),    # Thumb
        (8, 9), (9, 10), (10, 11),         # Index finger
        (12, 13), (13, 14), (14, 15),      # Middle finger
        (16, 17), (17, 18), (18, 19),      # Ring finger
        (20, 21), (21, 22), (22, 23)       # Pinky finger
    ]

    items_to_draw = []
    
    if draw_skeleton:
        for i, j in skeleton:
            # Skip if either point is NaN
            if np.isnan(joints_image[i]).any() or np.isnan(joints_image[j]).any():
                continue
            pt1 = joints_image[i]
            pt2 = joints_image[j]
            items_to_draw.append([
                (pt1[:2].astype(np.int32), pt2[:2].astype(np.int32)), 
                'line', 
                max(joints_camera[i, 2], joints_camera[j, 2])
            ])
    
    for i in range(len(joints_image)):
        # Skip if point is NaN
        if np.isnan(joints_image[i]).any():
            continue
        pt = joints_image[i]
        color_depth = joints_depth[i]
        radius = 7 if i < 3 else 5
        items_to_draw.append([
            (pt[:2].astype(np.int32), radius, i), 
            'point', 
            joints_camera[i, 2]
        ])
        
    if boundary:
        valid_joints = ~np.isnan(joints_camera).any(axis=1)
        if valid_joints.any():
            boundary = np.concatenate([
                joints_camera[valid_joints].min(0), 
                joints_camera[valid_joints].max(0)
            ], 0)
            boundary_line = get_boundary(boundary)
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
            if colors is not None:
                # Use colors array for joint coloring
                color = colors[item[0][2]]  # item[0][2] contains the joint index
                cv2.circle(image, tuple(item[0][0]), item[0][1], color, -1)
            else:
                # Use depth-based coloring as before
                color = (0, 0, int(255 * item[2]))
                cv2.circle(image, tuple(item[0][0]), item[0][1], color, -1)
        elif item[1] == 'boundary':
            cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), (0, 0, 0), 1)
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


def plot_hand_subplot(position, camera_params, joints_l, joints_r, title, colors=None):
    plt.subplot(position)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    img = plot_hand_camera(img, joints_l, **camera_params, boundary=True, colors=colors[0])
    img = plot_hand_camera(img, joints_r, **camera_params, boundary=True, colors=colors[1])
    plt.imshow(img)
    plt.title(title)
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.xticks([]); plt.yticks([])


def plot_hand_cv2(camera_params, joints_l, joints_r, colors=None):
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    img = plot_hand_camera(img, joints_l, **camera_params, boundary=True, colors=colors)
    img = plot_hand_camera(img, joints_r, **camera_params, boundary=True, colors=colors)
    return img

