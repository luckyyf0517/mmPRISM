import os
import glob
import numpy as np
from tqdm import tqdm
from glob import glob

def calculate_variance_for_npy_files():
    # Define the path pattern to match all required .npy files
    path_pattern = "/root/autodl-tmp/datasets/csl-news/poses/archive_*/*.npy"
    
    # Get all matching file paths
    file_paths = glob(path_pattern)
    
    # Check if any files were found
    if not file_paths:
        print("No .npy files found matching the pattern")
        return
    
    print(f"Found {len(file_paths)} .npy files")
    
    # Process each file and calculate variance
    for file_path in tqdm(file_paths):
        # Load the .npy file
        data = np.load(file_path)
        
        # Calculate variance
        variance = np.var(data)
        if variance > 0.1: 
            print(f"File: {file_path}, Variance: {variance}")
            
if __name__ == "__main__":
    calculate_variance_for_npy_files()
