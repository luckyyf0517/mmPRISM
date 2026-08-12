#!/usr/bin/env python3
"""
Script to load a random mmwave data sample from CollectedDailyDataset 
and process it with Processor to output the mmwave cube.
Then extract velocity argmax values and visualize as 3D matrix.
Can also use simulator to generate synthetic mmwave data from pose data.
"""

import os

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import CollectedSingleFrameDataset
from src.fmcw.simulator import Processor


def visualize_3d_cube(energy_matrix, velocity_matrix, title="Energy Cube Visualization", cube_index=0):
    """
    Visualize a 3D energy cube with energy-based transparency and velocity-based colors.
    Energy determines transparency: lower energy = more transparent, higher energy = more opaque.
    Velocity determines colors: different velocity values get different colors.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_alpha(0.0)  # Set figure background to transparent
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)  # Set axes background to transparent
    
    # Create a binary mask for non-zero values
    voxel_mask = energy_matrix > 0
    
    if not np.any(voxel_mask):
        print(f"No non-zero values found in energy cube {cube_index}!")
        return
    
    # Get non-zero indices using vectorized operations
    non_zero_indices = np.where(voxel_mask)
    
    # Extract coordinates for non-zero values
    x_coords = non_zero_indices[0]
    y_coords = non_zero_indices[1]
    z_coords = non_zero_indices[2] / 4  # Compress height to 1/4
    
    # Get energy values and velocity values for non-zero positions
    non_zero_energies = energy_matrix[voxel_mask]
    non_zero_velocities = velocity_matrix[voxel_mask]
    
    # Normalize energy values to 0-1 range for transparency mapping
    # Energy determines transparency: lower energy = more transparent, higher energy = more opaque
    normalized_energy = (non_zero_energies - non_zero_energies.min()) / (non_zero_energies.max() - non_zero_energies.min() + 1e-10)
    
    # Normalize velocity values to 0-1 range for color mapping
    # Velocity determines colors: different velocity values get different colors
    normalized_velocity = non_zero_velocities / 64

    # NOW draw the points with energy-based transparency and velocity-based colors
    # Use velocity values for color mapping, but set the lowest velocity (background) to gray
    colors = plt.cm.seismic(normalized_velocity)
    # Set background (lowest velocity) to gray
    gray = np.array([0.5, 0.5, 0.5, 1.0])  # RGBA for gray
    background_mask = normalized_velocity <= 0.01  # You can adjust threshold as needed
    colors[background_mask] = gray
    
    # Create alpha values based on energy: lower energy = more transparent
    alpha_values = 0.02 + 0.7 * normalized_energy  # Range from 0.1 to 0.8
    
    # Create size values based on energy: lower energy = smaller, higher energy = larger
    size_values = 80 + 70 * normalized_energy  # Range from 80 to 150
    
    # Apply alpha to colors by modifying the RGBA values
    # colors is already RGBA array, we just need to modify the alpha channel
    colors_with_alpha = colors.copy()
    colors_with_alpha[:, 3] = alpha_values  # Set alpha channel (4th column)
    
    # Draw all points in a single scatter plot with individual colors and sizes
    ax.scatter(x_coords, y_coords, z_coords, 
              c=colors_with_alpha, s=size_values, edgecolors='none')

    # Remove axis labels and grids for cleaner look
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    
    # Remove coordinate grids, planes, and ticks for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Remove axis ticks and tick lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Remove tick lines and axis lines
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)
    ax.tick_params(axis='z', which='both', length=0)
    
    # Remove axis lines completely
    ax.xaxis.line.set_color('none')
    ax.yaxis.line.set_color('none')
    ax.zaxis.line.set_color('none')
    
    # Set aspect ratio to 1:1:0.25 (X:Y:Z)
    ax.set_box_aspect([1, 1, 0.25])
    
    # Set viewing angle for more tilted perspective
    ax.view_init(elev=30, azim=-75)

    # Add custom axis labels with Times font and bold style
    # Azimuth: top center
    ax.text(38, 16, 0, 'Azimuth', fontsize=32, fontweight='bold', fontfamily='Times New Roman', ha='center')
    # Range: bottom left
    ax.text(16, -8, 0, 'Range', fontsize=32, fontweight='bold', fontfamily='Times New Roman', ha='left', va='bottom')
    # Elevate: right side (adjusted for compressed height)
    ax.text(32, 32, 4, 'Elevation', fontsize=32, fontweight='bold', fontfamily='Times New Roman', ha='left', va='center')

    # Draw semi-transparent gray faces on each side of the cube
    # Define the 8 vertices of the cube (height compressed to 1/4)
    # Note: boundaries should be from -0.5 to 31.5 to center on data points
    vertices = np.array([
        [-0.5, -0.5, 0], [31.5, -0.5, 0], [31.5, 31.5, 0], [-0.5, 31.5, 0],  # Bottom face
        [-0.5, -0.5, 7.75], [31.5, -0.5, 7.75], [31.5, 31.5, 7.75], [-0.5, 31.5, 7.75]  # Top face
    ])
    
    # Define the 6 faces using vertex indices
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [0, 3, 7, 4],  # Left face
        [1, 2, 6, 5]   # Right face
    ]
    
    # Draw each face as simple rectangular planes with semi-transparent gray color
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    for face in faces:
        face_vertices = vertices[face]
        # Create a simple rectangular polygon
        poly = Poly3DCollection([face_vertices], alpha=0.05, facecolor='gray', edgecolor='none')
        ax.add_collection3d(poly)
    
    # Draw cross-section lines at half height using dark green dashed lines, width 3
    cross_section_height = 7.75 / 2
    ax.plot([-0.5, 31.5], [-0.5, -0.5], [cross_section_height, cross_section_height], linestyle='--', color='darkgreen', linewidth=3, zorder=999)
    ax.plot([-0.5, 31.5], [31.5, 31.5], [cross_section_height, cross_section_height], linestyle='--', color='darkgreen', linewidth=3, zorder=999)
    ax.plot([-0.5, -0.5], [-0.5, 31.5], [cross_section_height, cross_section_height], linestyle='--', color='darkgreen', linewidth=3, zorder=999)
    ax.plot([31.5, 31.5], [-0.5, 31.5], [cross_section_height, cross_section_height], linestyle='--', color='darkgreen', linewidth=3, zorder=999)

    # Create the 8 vertices of the cube (height compressed to 1/4)
    # Boundaries from -0.5 to 31.5 to center on data points
    x = [-0.5, 31.5, 31.5, -0.5, -0.5, 31.5, 31.5, -0.5]
    y = [-0.5, -0.5, 31.5, 31.5, -0.5, -0.5, 31.5, 31.5]
    z = [0, 0, 0, 0, 7.75, 7.75, 7.75, 7.75]  # 31/4 = 7.75
    
    # Draw the 12 edges of the cube with higher zorder to appear on top
    # Bottom face edges
    ax.plot([-0.5, 31.5], [-0.5, -0.5], [0, 0], 'k-', linewidth=4, zorder=1000)
    ax.plot([31.5, 31.5], [-0.5, 31.5], [0, 0], 'k-', linewidth=4, zorder=1000)
    ax.plot([31.5, -0.5], [31.5, 31.5], [0, 0], 'k-', linewidth=4, zorder=1000)
    ax.plot([-0.5, -0.5], [31.5, -0.5], [0, 0], 'k-', linewidth=4, zorder=1000)
    
    # Top face edges
    ax.plot([-0.5, 31.5], [-0.5, -0.5], [7.75, 7.75], 'k-', linewidth=4, zorder=1000)
    ax.plot([31.5, 31.5], [-0.5, 31.5], [7.75, 7.75], 'k-', linewidth=4, zorder=1000)
    ax.plot([31.5, -0.5], [31.5, 31.5], [7.75, 7.75], 'k-', linewidth=4, zorder=1000)
    ax.plot([-0.5, -0.5], [31.5, -0.5], [7.75, 7.75], 'k-', linewidth=4, zorder=1000)
    
    # Vertical edges
    ax.plot([-0.5, -0.5], [-0.5, -0.5], [0, 7.75], 'k-', linewidth=4, zorder=1000)
    ax.plot([31.5, 31.5], [-0.5, -0.5], [0, 7.75], 'k-', linewidth=4, zorder=1000)
    ax.plot([31.5, 31.5], [31.5, 31.5], [0, 7.75], 'k-', linewidth=4, zorder=1000)
    ax.plot([-0.5, -0.5], [31.5, 31.5], [0, 7.75], 'k-', linewidth=4, zorder=1000)
    
    # Save the 3D cube visualization
    plt.tight_layout()
    plt.savefig(f"outputs/energy_cube_{cube_index}_{title}.png", dpi=200)
    plt.close()

    # Extract cross-section data at half height (z=16) for 2D visualization
    cross_section_z = 16  # Half of 32
    # cross_section_energy = energy_matrix[:, :, cross_section_z]
    # cross_section_velocity = velocity_matrix[:, :, cross_section_z]

    cross_section_energy = energy_matrix.sum(-1)
    cross_section_velocity = velocity_matrix[:, :, cross_section_z]
    
    return cross_section_energy, cross_section_velocity  # Return cross-section data for 2D plot


def visualize_2d_cross_section(cross_section_energy, cross_section_velocity, title="Cross-Section Visualization", cube_index=0):
    """
    Visualize a 2D cross-section with energy-based transparency and velocity-based colors.
    """
    # Normalize cross-section data
    cross_section_normalized_energy = (cross_section_energy - cross_section_energy.min()) / (cross_section_energy.max() - cross_section_energy.min() + 1e-10)
    cross_section_normalized_velocity = cross_section_velocity / 64
    
    # Only distinguish velocity sign: positive (>=0) or negative (<0)
    velocity_sign = (cross_section_velocity < 32).astype(float)  # 0 for negative, 1 for positive
    cross_section_colors = plt.cm.seismic(velocity_sign)
    gray = np.array([0.5, 0.5, 0.5, 1.0])
    cross_section_background_mask = cross_section_normalized_velocity <= 0.01
    cross_section_colors[cross_section_background_mask] = gray
    
    # Apply energy-based transparency
    cross_section_alpha = 0.1 + 0.9 * cross_section_normalized_energy
    cross_section_colors[:, :, 3] = cross_section_alpha
    
    # Create 2D cross-section plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D cross-section visualization
    im = ax.imshow(cross_section_colors, extent=[-0.5, 31.5, -0.5, 31.5], origin='lower')
    # Set labels with Times New Roman and bold, and remove ticks
    # Save the 2D cross-section
    plt.tight_layout()
    plt.axis('off')
    plt.savefig(f"outputs/range_angle_cross_section_{cube_index}_{title}.png", 
               dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_angle_doppler(mmwave_data_4d, title="Angle-Doppler Visualization", cube_index=0):
    """
    Visualize angle-doppler plot with energy-based transparency and velocity-based colors.
    Energy determines transparency: lower energy = more transparent, higher energy = more opaque.
    Velocity determines colors: >32 = blue, <32 = red.
    Uses mmwave_data_4d[:, sum, :, 16] slice.
    """
    # Extract angle-doppler data: [:, sum, :, 16] -> [64, 32]
    # Sum across range dimension (axis 2), take elevation slice at 16
    angle_doppler_data = mmwave_data_4d.sum(dim=1).sum(dim=-1)  # [64, 32]

    # Convert to numpy for processing
    angle_doppler_data_np = angle_doppler_data.numpy()  # [64, 32]
    
    # Normalize energy data (each pixel has its own energy value)
    angle_doppler_normalized_energy = (angle_doppler_data_np - angle_doppler_data_np.min()) / (angle_doppler_data_np.max() - angle_doppler_data_np.min() + 1e-10)

    # Get velocity argmax for each pixel to determine color mapping
    velocity_argmax = angle_doppler_data.argmax(dim=0).numpy()  # [32] - argmax along velocity dimension for each angle
    
    # Create velocity-based colors with energy-based transparency
    velocity_colors = np.zeros((64, 32, 4))  # RGBA array for 2D data
    
    # Make red and blue colors deeper by increasing the scaling factor
    blue_scale = 4.5  # Increase for deeper blue
    red_scale = 4.5   # Increase for deeper red

    # Apply color mapping based on velocity threshold for each angle column
    for i in range(32):  # For each angle
        if velocity_argmax[i] > 32:
            # Use blue colormap for velocities > 32
            # Apply threshold: only show color for energy > 0.1, make colors deeper
            energy_threshold = angle_doppler_normalized_energy[:, i] > 0.1
            blue_colors = plt.cm.Blues(np.clip(angle_doppler_normalized_energy[:, i] * blue_scale, 0, 1))  # Make colors deeper
            blue_colors[~energy_threshold] = [0, 0, 0, 0]  # Transparent for low energy
            velocity_colors[:, i, :3] = blue_colors[:, :3]
        else:
            # Use red colormap for velocities <= 32
            # Apply threshold: only show color for energy > 0.1, make colors deeper
            energy_threshold = angle_doppler_normalized_energy[:, i] > 0.1
            red_colors = plt.cm.Reds(np.clip(angle_doppler_normalized_energy[:, i] * red_scale, 0, 1))  # Make colors deeper
            red_colors[~energy_threshold] = [0, 0, 0, 0]  # Transparent for low energy
            velocity_colors[:, i, :3] = red_colors[:, :3]
    
    # Apply energy-based transparency for each pixel
    # Make low energy areas more transparent, high energy areas more opaque
    angle_doppler_alpha = np.where(angle_doppler_normalized_energy > 0.1, 
                                  0.3 + 0.7 * angle_doppler_normalized_energy, 
                                  0.0)  # Completely transparent for low energy
    velocity_colors[:, :, 3] = angle_doppler_alpha
    
    # Create 2D angle-doppler plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create angle-doppler visualization (2D data) with 1:1 aspect ratio (square pixels)
    im = ax.imshow(velocity_colors, origin='lower', aspect='auto')
    ax.set_aspect(1)  # Force 1:1 aspect ratio (square)
    plt.axis('off')
    
    # Save the angle-doppler plot
    plt.tight_layout()
    plt.savefig(f"outputs/angle_doppler_{cube_index}_{title}.png", dpi=200)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process mmwave data with option to use simulator')
    parser.add_argument('--use-simulator', action='store_true', 
                       help='Use simulator to generate synthetic mmwave data from pose data')

    args = parser.parse_args()
    
    use_simulator = args.use_simulator
    
    print(f"Mode: {'Simulator' if use_simulator else 'Real Data'}")
    
    # Configuration for the dataset
    opt = {
        "load_feature": False,
        "norm_pose": True,
        "modalities": {
            "use_pred_pose": False,
            "use_gt_pose": True,  # Always load pose data for simulator
            "use_raw_pose": False,
            "use_features": False,
            "use_mmwave": not use_simulator,  # Only load mmwave if not using simulator
        }
    }
    
    # Path to the dataset split (you may need to adjust this path)
    split_path = 'dataset/collected-200/train.json'
    
    # Check if the split file exists
    if not os.path.exists(split_path):
        print(f"Dataset split file not found: {split_path}")
        print("Please make sure the dataset is properly set up.")
        return
    
    # Initialize dataset
    dataset = CollectedSingleFrameDataset(opt, split_path=split_path)
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    # Load a random sample
    random_index = np.random.randint(0, len(dataset))
    random_index = 16012
    print(f"Loading sample at index {random_index}...")
    
    try:
        sample = dataset[random_index]
        print("Sample loaded successfully!")
        
        # Print sample information
        print(f"Sample ID: {sample['id']}")
        print(f"Frame index: {sample['frame_idx']}")
        
        if use_simulator:
            # Use simulator to generate mmwave data
            print("Using simulator to generate mmwave data...")

            pose_data = sample['joints']
            print(f"Pose data shape: {pose_data.shape}")
            
            # Extract 3D points and velocities from pose data
            points_3d = torch.from_numpy(pose_data).unsqueeze(0)  # Add batch dimension
            velocities_3d = torch.zeros_like(points_3d)
            velocities_3d[0, 0] = -1  # Left side velocity
            velocities_3d[0, 1] = 1   # Right side velocity
            
            print(f"Simulator input - Points shape: {points_3d.shape}, Velocities shape: {velocities_3d.shape}")
            
            # Initialize simulator
            from src.fmcw.simulator import Simulation
            simulator = Simulation()
            
            # Generate raw radar frame using simulator
            with torch.no_grad():
                raw_radar_frame = simulator(points_3d, velocities_3d)
            
            print(f"Simulator output shape: {raw_radar_frame.shape}")
            
            # Process with Processor
            print("Processing simulator data with Processor...")
            processor = Processor(learnable_weights=False)
            processor.if_process_range = True
            
            with torch.no_grad():
                mmwave_cube = processor(raw_radar_frame)
            
            print(f"Processor cube shape: {mmwave_cube.shape}")
            print(f"Processor cube dtype: {mmwave_cube.dtype}")
            
        else:
            # Use real mmwave data from dataset
            print("Using real mmwave data from dataset...")
            
            # Check if mmwave data is available
            if 'mmwave' not in sample:
                print("No mmwave data found in the sample!")
                return
                
            mmwave_data = sample['mmwave']
            print(f"Raw mmwave data shape: {mmwave_data.shape}")
            print(f"Raw mmwave data type: {mmwave_data.dtype}")
            
            # Convert to tensor and add batch dimension
            mmwave_tensor = torch.from_numpy(mmwave_data).unsqueeze(0)
            print(f"mmwave tensor shape: {mmwave_tensor.shape}")
            
            # Process with default beamforming weights
            print("Processing with default beamforming weights...")
            processor = Processor(learnable_weights=False)
            
            with torch.no_grad():
                mmwave_cube = processor(mmwave_tensor)
            
            print(f"Processor cube shape: {mmwave_cube.shape}")
            print(f"Processor cube dtype: {mmwave_cube.dtype}")
        
        # Extract the four dimensions: [B, 64, 32, 32, 32]
        # We want to visualize velocity argmax values
        mmwave_data_4d = mmwave_cube.squeeze(0)  # [64, 32, 32, 32]

        # Compute energy by summing across the first dimension (64 velocity bins)
        print("Computing energy by summing across velocity dimensions...")
        energy_matrix = torch.sum(mmwave_data_4d, dim=0).numpy()  # [32, 32, 32]
        
        # Get velocity argmax for color mapping
        print("Computing velocity argmax for color mapping...")
        velocity_matrix = torch.argmax(mmwave_data_4d, dim=0).numpy()  # [32, 32, 32]
        
        # Visualize the energy cube with velocity-based colors
        print("Creating 3D visualization for energy with velocity colors...")
        cross_section_energy, cross_section_velocity = visualize_3d_cube(energy_matrix, velocity_matrix, f"Energy_Velocity_Sample", "energy_velocity")
        
        # Create 2D cross-section visualization
        print("Creating 2D cross-section visualization...")
        visualize_2d_cross_section(cross_section_energy, cross_section_velocity, f"Energy_Velocity_Sample", "energy_velocity")
        
        # Create angle-doppler visualization
        print("Creating angle-doppler visualization...")
        visualize_angle_doppler(mmwave_data_4d, f"Energy_Velocity_Sample", "energy_velocity")

        print("Processing completed successfully!")

        
    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Usage:
# python view_mmwave_cube.py                    # Use real mmwave data (default)
# python view_mmwave_cube.py --use-simulator   # Use simulator to generate synthetic data