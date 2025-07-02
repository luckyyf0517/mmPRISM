import os
import numpy as np
from tqdm import tqdm

# Get all mmwave files in collected_base/mmwave
mmwave_dir = 'data/collected_base/mmwave'
mmwave_files = [f for f in os.listdir(mmwave_dir) if f.endswith('.npy')]

# Process each mmwave file
for mmwave_file in tqdm(mmwave_files, desc="Processing mmwave files"):
    # Load mmwave data
    mmwave_path = os.path.join(mmwave_dir, mmwave_file)
    mmwave_data = np.load(mmwave_path)
    
    # Get sequence ID (filename without extension)
    seq_id = os.path.splitext(mmwave_file)[0]
    
    # Create sequence directory
    seq_dir = os.path.join(mmwave_dir, seq_id)
    os.makedirs(seq_dir, exist_ok=True)
    
    # Save each frame as separate file
    for frame_idx in range(mmwave_data.shape[0]):
        frame_filename = f"{frame_idx:04d}.npy"
        frame_path = os.path.join(seq_dir, frame_filename)
        np.save(frame_path, mmwave_data[frame_idx])
        
    # Remove original file after splitting
    os.remove(mmwave_path)
