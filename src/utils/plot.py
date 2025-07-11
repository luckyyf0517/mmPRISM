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


def plot_hand_camera_beautiful(image, joints, camera_position, camera_intrinsic, boundary=False, draw_skeleton=True):
    """Beautiful hand keypoint visualization with enhanced colors and effects"""
    fx, fy, cx, cy = camera_intrinsic 
    
    # Calculate positions relative to camera
    joints_camera = joints - camera_position
    
    # Project to image plane
    joints_image = joints_camera.copy()
    joints_image[..., 0] = joints_image[..., 0] / joints_image[..., 2] * fx + cx
    joints_image[..., 1] = joints_image[..., 1] / joints_image[..., 2] * fy + cy
    
    # Calculate depth values for coloring
    valid_joints = ~np.isnan(joints_camera[:, 2])
    joints_depth = np.zeros_like(joints_camera[:, 2])
    if valid_joints.any():
        joints_depth[valid_joints] = 1 - (joints_camera[valid_joints, 2] - joints_camera[valid_joints, 2].min()) / (joints_camera[valid_joints, 2].max() - joints_camera[valid_joints, 2].min())

    # Enhanced skeleton connection definitions with colors
    skeleton_connections = {
        'arm': [(0, 1), (1, 2)],  # Arm connections
        'palm': [(2, 3), (3, 8), (8, 12), (12, 16), (16, 20), (20, 3)],  # Palm connections
        'thumb': [(3, 4), (4, 5), (5, 6), (6, 7)],    # Thumb
        'index': [(8, 9), (9, 10), (10, 11)],         # Index finger
        'middle': [(12, 13), (13, 14), (14, 15)],     # Middle finger
        'ring': [(16, 17), (17, 18), (18, 19)],       # Ring finger
        'pinky': [(20, 21), (21, 22), (22, 23)]       # Pinky finger
    }
    
    # Define beautiful colors for different parts
    colors = {
        'arm': (180, 60, 60),        # Deep red - arm
        'palm': (60, 180, 60),       # Deep green - palm
        'thumb': (180, 120, 60),     # Deep orange - thumb
        'index': (60, 90, 180),      # Deep blue - index
        'middle': (180, 60, 180),    # Deep purple - middle
        'ring': (60, 180, 180),      # Deep cyan - ring
        'pinky': (180, 180, 60)      # Deep yellow - pinky
    }
    # Joint type colors
    joint_colors = {
        'arm': (220, 80, 80),
        'palm': (80, 220, 80),
        'finger': (80, 120, 220)
    }

    items_to_draw = []
    
    # Draw boundary first (furthest back)
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
                    max(pt1[2], pt2[2]),
                    (150, 150, 150),  # Gray color for boundary
                    1
                ])
    
    if draw_skeleton:
        # Draw skeleton connections with different colors
        for part_name, connections in skeleton_connections.items():
            color = colors[part_name]
            for i, j in connections:
                # Skip if either point is NaN
                if np.isnan(joints_image[i]).any() or np.isnan(joints_image[j]).any():
                    continue
                pt1 = joints_image[i]
                pt2 = joints_image[j]
                
                # Calculate line thickness based on depth and part type
                thickness = 5 if part_name == 'arm' else 4
                
                items_to_draw.append([
                    (pt1[:2].astype(np.int32), pt2[:2].astype(np.int32)), 
                    'line', 
                    max(joints_camera[i, 2], joints_camera[j, 2]),
                    color,
                    thickness
                ])
    
    # Draw joints with enhanced styling
    for i in range(len(joints_image)):
        # Skip if point is NaN
        if np.isnan(joints_image[i]).any():
            continue
            
        pt = joints_image[i]
        depth_ratio = joints_depth[i] if i < len(joints_depth) else 0.5
        
        # Determine joint type and color
        if i < 3:  # Arm joints
            color = joint_colors['arm']
            # radius = 8
        elif i == 3:  # Palm center
            color = joint_colors['palm']
            # radius = 7
        else:  # Finger joints
            color = joint_colors['finger']
            # radius = 6
        
        radius = 2

        # Add depth-based brightness variation
        brightness = 0.7 + 0.3 * depth_ratio
        enhanced_color = tuple(int(c * brightness) for c in color)
        
        items_to_draw.append([
            (pt[:2].astype(np.int32), radius), 
            'point', 
            joints_camera[i, 2],
            enhanced_color,
            i  # joint index for special handling
        ])
    
    # Sort by z-depth for proper layering (larger z values drawn first - further back)
    items_to_draw.sort(key=lambda x: -x[2])
    
    # Draw items with enhanced effects
    for item in items_to_draw:
        if item[1] == 'line':
            pt1, pt2 = item[0]
            color = item[3]
            thickness = item[4]
            
            # Draw main line with dashed effect
            dash_length = 10
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                # Calculate number of dashes based on distance
                num_dashes = max(2, int(dist / (2 * dash_length)))
                
                # # Draw dashed line segments
                # for i in range(num_dashes):
                #     start_ratio = i / num_dashes
                #     end_ratio = (i + 0.5) / num_dashes  # Only draw half of each segment for dashed effect
                    
                #     start_x = int(pt1[0] + dx * start_ratio)
                #     start_y = int(pt1[1] + dy * start_ratio)
                #     end_x = int(pt1[0] + dx * end_ratio)
                #     end_y = int(pt1[1] + dy * end_ratio)
                    
                #     cv2.line(image, 
                #             (start_x, start_y), 
                #             (end_x, end_y), 
                #             color, thickness,
                #             lineType=cv2.LINE_AA)

                cv2.line(image, tuple(pt1), tuple(pt2), color, thickness, lineType=cv2.LINE_AA)
            
        elif item[1] == 'point':
            center, radius = item[0]
            color = item[3]
            joint_idx = item[4]
            
            # Draw shadow
            shadow_offset = 2
            shadow_color = (50, 50, 50)
            cv2.circle(image, 
                      (center[0] + shadow_offset, center[1] + shadow_offset), 
                      radius, shadow_color, -1)
            
            # Draw outer ring
            outer_color = tuple(max(0, c - 40) for c in color)
            cv2.circle(image, tuple(center), radius + 1, outer_color, 2)
            
            # Draw main circle
            cv2.circle(image, tuple(center), radius, color, -1)
            
            # Draw inner highlight
            highlight_color = tuple(min(255, c + 60) for c in color)
            cv2.circle(image, 
                      (center[0] - 1, center[1] - 1), 
                      max(1, radius - 2), highlight_color, -1)
            
        elif item[1] == 'boundary':
            pt1, pt2 = item[0]
            color = item[3]
            thickness = item[4]
            cv2.line(image, tuple(pt1), tuple(pt2), color, thickness)
    
    return image
