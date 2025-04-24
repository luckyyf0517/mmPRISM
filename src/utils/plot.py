import os
import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_hand_camera(image, joints, camera_position, camera_intrinsic, boundary=False, draw_skeleton=True):
    fx, fy, cx, cy = camera_intrinsic 
    
    # 只需要计算相对于相机的位置
    joints_camera = joints - camera_position
    
    # 投影到图像平面
    joints_image = joints_camera.copy()
    joints_image[..., 0] = joints_image[..., 0] / joints_image[..., 2] * fx + cx
    joints_image[..., 1] = joints_image[..., 1] / joints_image[..., 2] * fy + cy
    # 计算深度值用于着色
    joints_depth = 1 - (joints_camera[..., 2] - joints_camera[:, 2].min()) / (joints_camera[:, 2].max() - joints_camera[:, 2].min())

    # 骨架连接定义
    skeleton = [
        # 手臂骨架 (0,1,2)
        (0, 1), (1, 2),
        # 手部骨架 (从索引3开始，对应原来的0)
        (2, 3),  # 手臂末端连接到手掌
        (3, 8), (8, 12), (12, 16), (16, 20), (20, 3),  # 手掌
        (3, 4), (4, 5), (5, 6), (6, 7),    # 拇指
        (8, 9), (9, 10), (10, 11),         # 食指
        (12, 13), (13, 14), (14, 15),      # 中指
        (16, 17), (17, 18), (18, 19),      # 无名指
        (20, 21), (21, 22), (22, 23)       # 小指
    ]
    
    items_to_draw = []
    
    if draw_skeleton:
        for i, j in skeleton:
            pt1 = joints_image[i]
            pt2 = joints_image[j]
            items_to_draw.append([
                (pt1[:2].astype(np.int32), pt2[:2].astype(np.int32)), 
                'line', 
                max(joints_camera[i, 2], joints_camera[j, 2])
            ])
    
    for i in range(len(joints_image)):
        pt = joints_image[i]
        color_depth = joints_depth[i]
        # 为手臂关节点使用不同的大小
        radius = 7 if i < 3 else 5
        items_to_draw.append([
            (pt[:2].astype(np.int32), radius), 
            'point', 
            joints_camera[i, 2]
        ])
        
    if boundary:
        boundary = np.concatenate([joints_camera.min(0), joints_camera.max(0)], 0)
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
