import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_hand_camera(image, joints, camera_position, camera_intrinsic, boundary=False, draw_skeleton=True, colors=None, skeleton_thickness=2, joint_size=None, enhanced_visual=True):
    # For orthographic projection, treat camera_intrinsic as [scale_x, scale_y, offset_x, offset_y]
    scale_x, scale_y, offset_x, offset_y = camera_intrinsic 
    
    # Calculate positions relative to camera
    joints_camera = joints - camera_position
    
    # Project to image plane (orthographic projection)
    joints_image = joints_camera.copy()
    joints_image[..., 0] = joints_image[..., 0] * scale_x + offset_x
    joints_image[..., 1] = joints_image[..., 1] * scale_y + offset_y
    # Calculate depth values for shading
    valid_joints = ~np.isnan(joints_camera[:, 2])
    joints_depth = np.zeros_like(joints_camera[:, 2])
    if valid_joints.any():
        joints_depth[valid_joints] = 1 - (joints_camera[valid_joints, 2] - joints_camera[valid_joints, 2].min()) / (joints_camera[valid_joints, 2].max() - joints_camera[valid_joints, 2].min())

    # Enhanced color scheme for better visual appeal
    if enhanced_visual:
        # Create a more appealing color palette
        base_colors = [
            (0, 0, 0),         # Black for arm joints
            (0, 0, 0),         # Black for arm joints
            (0, 0, 0),         # Black for arm joints
            (0, 0, 0),         # Black for palm center
            (0, 0, 0),         # Black for palm
            (0, 0, 0),         # Black for palm
            (0, 0, 0),         # Black for thumb
            (0, 0, 0),         # Black for thumb
            (0, 0, 0),         # Black for index
            (0, 0, 0),         # Black for index
            (0, 0, 0),         # Black for index
            (0, 0, 0),         # Black for middle
            (0, 0, 0),         # Black for middle
            (0, 0, 0),         # Black for middle
            (0, 0, 0),         # Black for ring
            (0, 0, 0),         # Black for ring
            (0, 0, 0),         # Black for ring
            (0, 0, 0),         # Black for pinky
            (0, 0, 0),         # Black for pinky
            (0, 0, 0),         # Black for pinky
        ]
        
        # Ensure we have enough colors for all joints
        num_joints = len(joints)
        if num_joints > len(base_colors):
            # Extend color palette with additional colors
            additional_colors = [
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
                (0, 0, 0),         # Black for additional joints
            ]
            # Repeat colors if needed
            while len(base_colors) < num_joints:
                base_colors.extend(additional_colors)
        
        # Apply depth-based color variation
        enhanced_colors = []
        for i in range(num_joints):
            if i < len(base_colors):
                color = base_colors[i]
            else:
                # Fallback color for any remaining joints
                color = (0, 0, 0)
            
            # Keep joints black regardless of depth
            enhanced_colors.append(color)
        
        colors = enhanced_colors

    # Normalize colors to 0-1 range if provided (for backward compatibility)
    if colors is not None and not enhanced_visual:
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
                max(joints_camera[i, 2], joints_camera[j, 2]),
                (i, j)  # Store joint indices for color selection
            ])
    
    for i in range(len(joints_image)):
        # Skip if point is NaN
        if np.isnan(joints_image[i]).any():
            continue
        pt = joints_image[i]
        color_depth = joints_depth[i]
        # Use custom joint size if provided, otherwise use default sizes
        if joint_size is not None:
            radius = joint_size
        else:
            radius = 8 if i < 3 else 6  # Slightly larger default sizes
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
            boundary_line[..., 0] = boundary_line[..., 0] * scale_x + offset_x
            boundary_line[..., 1] = boundary_line[..., 1] * scale_y + offset_y
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
    
    # Separate items by type for proper layering
    arm_lines = []
    palm_lines = []
    other_items = []
    
    for item in items_to_draw:
        if item[1] == 'line' and len(item) > 3:
            i, j = item[3]
            # Check if this is an arm line (both joints < 3)
            if i < 3 and j < 3:
                arm_lines.append(item)
            else:
                palm_lines.append(item)
        else:
            other_items.append(item)
    
    # Draw items in proper order: arm lines first, then other items, then palm lines
    all_items = arm_lines + other_items + palm_lines
    
    # Draw items with enhanced visual effects
    for item in all_items:
        if item[1] == 'line':
            if enhanced_visual and len(item) > 3:
                # Enhanced skeleton lines with specific colors for different parts
                i, j = item[3]
                
                # Determine line color based on joint indices
                if i < 3 and j < 3:
                    # Arm lines (joints 0, 1, 2) - use lighter red
                    line_color = (200, 60, 60)  # Lighter red
                elif i >= 3 or j >= 3:
                    # Palm and finger lines - use lighter blue
                    line_color = (60, 100, 200)  # Lighter blue
                else:
                    # Fallback color
                    line_color = (0, 0, 0)
                
                # Draw shadow first
                shadow_offset = 1
                cv2.line(image, 
                        (item[0][0][0] + shadow_offset, item[0][0][1] + shadow_offset),
                        (item[0][1][0] + shadow_offset, item[0][1][1] + shadow_offset),
                        (50, 50, 50), skeleton_thickness + 1)
                
                # Draw main line with determined color
                cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), line_color, skeleton_thickness)
                
            else:
                cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), (0, 0, 0), skeleton_thickness)
                
        elif item[1] == 'point':
            if enhanced_visual and colors is not None and item[0][2] < len(colors):
                # Enhanced joint visualization with shadow and highlight
                color = colors[item[0][2]]
                radius = item[0][1]
                
                # Draw shadow
                shadow_offset = 2
                shadow_radius = radius + 1
                cv2.circle(image, 
                          (item[0][0][0] + shadow_offset, item[0][0][1] + shadow_offset),
                          shadow_radius, (30, 30, 30), -1)
                
                # Draw main joint
                cv2.circle(image, tuple(item[0][0]), radius, color, -1)
                
                # Draw highlight for 3D effect
                highlight_radius = max(2, radius // 3)
                highlight_offset = max(1, radius // 4)
                highlight_color = tuple(min(255, int(c * 1.3)) for c in color)
                cv2.circle(image, 
                          (item[0][0][0] - highlight_offset, item[0][0][1] - highlight_offset),
                          highlight_radius, highlight_color, -1)
                
            elif colors is not None and item[0][2] < len(colors):
                # Use colors array for joint coloring (with bounds check)
                color = colors[item[0][2]]  # item[0][2] contains the joint index
                cv2.circle(image, tuple(item[0][0]), item[0][1], color, -1)
            else:
                # Use depth-based coloring as before
                color = (0, 0, int(255 * item[2]))
                cv2.circle(image, tuple(item[0][0]), item[0][1], color, -1)
                
        elif item[1] == 'boundary':
            if enhanced_visual:
                # Enhanced boundary with subtle styling
                cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), (100, 100, 100), 2)
                # Draw a thinner highlight line
                cv2.line(image, tuple(item[0][0]), tuple(item[0][1]), (200, 200, 200), 1)
            else:
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


def plot_hand_subplot(position, camera_params, joints_l, joints_r, title, enhanced_visual=False):
    plt.subplot(position)
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    img = plot_hand_camera(img, joints_l, **camera_params, boundary=True, enhanced_visual=enhanced_visual)
    img = plot_hand_camera(img, joints_r, **camera_params, boundary=True, enhanced_visual=enhanced_visual)
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
