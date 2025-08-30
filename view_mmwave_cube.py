#!/usr/bin/env python3
"""
Script to load a random mmwave data sample from CollectedDailyDataset 
and process it with Processor to output the mmwave cube.
Then extract velocity argmax values and visualize as 3D matrix.
Can also use simulator to generate synthetic mmwave data from pose data.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.data.dataset import CollectedSingleFrameDataset
from src.fmcw.simulator import Processor

def draw_colorbar(vmax=64, save_path="outputs/velocity_colorbar.pdf"):
    """
    Draw a horizontal colorbar for velocity from -vmax to vmax.
    Save as PDF.
    """
    # Create figure and remove default axes
    fig = plt.figure(figsize=(8, 1.0))
    fig.clear()  # Remove any default axes
    
    # Create colorbar directly
    norm = plt.Normalize(-vmax, vmax)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    
    # Add colorbar to figure with specific position
    cbar_ax = fig.add_axes([0.1, 0.2, 0.8, 0.3])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Set colorbar ticks and labels
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels(['-Vmax', '0', 'Vmax'])
    cbar.ax.tick_params(labelsize=20, width=2)  # Make tick marks thicker and labels larger
    
    # Make font bold and use Times font
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('Times New Roman')
    
    # Make colorbar border thicker
    cbar.outline.set_linewidth(2)
    
    # Transparent background
    fig.patch.set_alpha(0.0)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Colorbar saved to: {save_path}")

def visualize_energy_cube(energy_matrix, velocity_matrix, title="Energy Cube Visualization", cube_index=0):
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
    
    # Add custom axis labels with Times font and bold style
    # Azimuth: top center
    ax.text(38, 16, 0, 'Azimuth', fontsize=32, fontweight='bold', fontfamily='Times New Roman', ha='center')
    # Range: bottom left
    ax.text(16, -8, 0, 'Range', fontsize=32, fontweight='bold', fontfamily='Times New Roman', ha='left', va='bottom')
    # Elevate: right side (adjusted for compressed height)
    ax.text(32, 32, 4, 'Elevation', fontsize=32, fontweight='bold', fontfamily='Times New Roman', ha='left', va='center')

    # NOW draw the black wireframe box LAST (on top of everything)
    # Create the 8 vertices of the cube (height compressed to 1/4)
    x = [0, 31, 31, 0, 0, 31, 31, 0]
    y = [0, 0, 31, 31, 0, 0, 31, 31]
    z = [0, 0, 0, 0, 7.75, 7.75, 7.75, 7.75]  # 31/4 = 7.75
    
    # Draw the 12 edges of the cube with higher zorder to appear on top
    # Bottom face edges
    ax.plot([0, 31], [0, 0], [0, 0], 'k-', linewidth=3, zorder=1000)
    ax.plot([31, 31], [0, 31], [0, 0], 'k-', linewidth=3, zorder=1000)
    ax.plot([31, 0], [31, 31], [0, 0], 'k-', linewidth=3, zorder=1000)
    ax.plot([0, 0], [31, 0], [0, 0], 'k-', linewidth=3, zorder=1000)
    
    # Top face edges
    ax.plot([0, 31], [0, 0], [7.75, 7.75], 'k-', linewidth=3, zorder=1000)
    ax.plot([31, 31], [0, 31], [7.75, 7.75], 'k-', linewidth=3, zorder=1000)
    ax.plot([31, 0], [31, 31], [7.75, 7.75], 'k-', linewidth=3, zorder=1000)
    ax.plot([0, 0], [31, 0], [7.75, 7.75], 'k-', linewidth=3, zorder=1000)
    
    # Vertical edges
    ax.plot([0, 0], [0, 0], [0, 7.75], 'k-', linewidth=3, zorder=1000)
    ax.plot([31, 31], [0, 0], [0, 7.75], 'k-', linewidth=3, zorder=1000)
    ax.plot([31, 31], [31, 31], [0, 7.75], 'k-', linewidth=3, zorder=1000)
    ax.plot([0, 0], [31, 31], [0, 7.75], 'k-', linewidth=3, zorder=1000)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"outputs/energy_cube_{cube_index}_{title}.png", dpi=200)
    plt.close()  # Close figure to free memory

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process mmwave data with option to use simulator')
    parser.add_argument('--use-simulator', action='store_true', 
                       help='Use simulator to generate synthetic mmwave data from pose data')
    parser.add_argument('--use-real-data', action='store_true',
                       help='Use real mmwave data from dataset (default)')
    args = parser.parse_args()
    
    # Determine whether to use simulator or real data
    use_simulator = args.use_simulator
    if not args.use_simulator and not args.use_real_data:
        use_simulator = False  # Default to real data
    
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
    
    # Load a specific sample
    random_index = np.random.randint(0, len(dataset))
    random_index = 3265
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
            # Assuming pose data has shape [T, 2, 24, 3] where:
            # T: time steps, 2: left/right sides, 24: body+hand points, 3: xyz coordinates
            points_3d = torch.from_numpy(pose_data)  # Take first time step [2, 24, 3]
            velocities_3d = torch.zeros_like(points_3d)  # Zero velocities for static pose
            
            # Add batch dimension for simulator
            points_3d = points_3d.unsqueeze(0)  # [1, 2, 24, 3]
            velocities_3d = velocities_3d.unsqueeze(0)  # [1, 2, 24, 3]
            velocities_3d[0, 0] = -1
            velocities_3d[0, 1] = 1
            
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
            
            # Convert numpy array to torch tensor and add batch dimension
            mmwave_tensor = torch.from_numpy(mmwave_data).unsqueeze(0)  # Add batch dimension
            print(f"mmwave tensor shape: {mmwave_tensor.shape}")
            
            # Process with DEFAULT beamforming weights
            print("Processing with DEFAULT beamforming weights...")
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
        visualize_energy_cube(energy_matrix, velocity_matrix, f"Energy_Velocity_Sample", "energy_velocity")
        
        # Create horizontal colorbar for velocity mapping
        print("Creating horizontal colorbar for velocity mapping...")
        draw_colorbar(vmax=64, save_path="outputs/velocity_colorbar.pdf")
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Usage examples:
# python load_and_process_mmwave.py                    # Use real mmwave data (default)
# python load_and_process_mmwave.py --use-real-data   # Use real mmwave data
# python load_and_process_mmwave.py --use-simulator   # Use simulator to generate synthetic data