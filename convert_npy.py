#!/usr/bin/env python3
"""
NPY Format Converter for mmWave Data

This script converts between two data formats:
1. Individual frame files: data/mmprism/collected_*/mmwave/seqid:%04d/frameid:%04d.npy
2. Combined sequence files: data/mmprism/collected_*/mmwave/seqid:%04d.npy

The script performs in-place conversion:
- frames_to_sequence: Combines frame files into sequence files and removes frame directories
- sequence_to_frames: Splits sequence files into frame directories and removes sequence files

Usage:
    python convert_npy.py --mode frames_to_sequence --input_dir data/mmprism/collected_demo/mmwave
    python convert_npy.py --mode sequence_to_frames --input_dir data/mmprism/collected_demo/mmwave
"""

import argparse
import logging
import os
import sys
import gc
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import glob
from tqdm import tqdm


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def find_sequence_dirs(input_dir: str) -> List[str]:
    """Find all sequence directories in the input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find directories that contain .npy files
    seq_dirs = []
    for item in input_path.iterdir():
        if item.is_dir():
            # Check if directory contains .npy files
            npy_files = list(item.glob("*.npy"))
            if npy_files:
                seq_dirs.append(str(item))
    
    seq_dirs.sort()
    logging.info(f"Found {len(seq_dirs)} sequence directories")
    return seq_dirs


def find_sequence_files(input_dir: str) -> List[str]:
    """Find all sequence .npy files in the input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    # Find .npy files directly in the directory (not in subdirectories)
    seq_files = []
    for npy_file in input_path.glob("*.npy"):
        seq_files.append(str(npy_file))
    
    seq_files.sort()
    logging.info(f"Found {len(seq_files)} sequence files")
    return seq_files


def get_frame_files(seq_dir: str) -> List[str]:
    """Get all frame files in a sequence directory, sorted by frame ID"""
    seq_path = Path(seq_dir)
    frame_files = []
    
    for npy_file in sorted(seq_path.glob("*.npy")):
        frame_files.append(str(npy_file))
    
    logging.debug(f"Found {len(frame_files)} frame files in {seq_dir}")
    return frame_files


def load_frame_data(frame_files: List[str]) -> np.ndarray:
    """Load and combine frame data into a single array"""
    if not frame_files:
        raise ValueError("No frame files provided")
    
    logging.debug(f"Loading {len(frame_files)} frame files...")
    
    # Load first frame to get shape and dtype
    first_frame = np.load(frame_files[0])
    logging.debug(f"First frame shape: {first_frame.shape}, dtype: {first_frame.dtype}")
    
    # Initialize combined array
    combined_shape = (len(frame_files),) + first_frame.shape
    combined_data = np.zeros(combined_shape, dtype=first_frame.dtype)
    
    # Load all frames
    for i, frame_file in enumerate(frame_files):
        try:
            frame_data = np.load(frame_file)
            if frame_data.shape != first_frame.shape:
                logging.warning(f"Frame {frame_file} has different shape: {frame_data.shape} vs {first_frame.shape}")
                continue
            combined_data[i] = frame_data
            logging.debug(f"Loaded frame {i+1}/{len(frame_files)}: {frame_file}")
        except Exception as e:
            logging.error(f"Failed to load frame {frame_file}: {e}")
            continue
    
    logging.debug(f"Successfully loaded {len(frame_files)} frames, combined shape: {combined_data.shape}")
    return combined_data


def save_sequence_data(seq_data: np.ndarray, output_path: str) -> None:
    """Save combined sequence data to file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, seq_data)
    logging.debug(f"Saved sequence data to: {output_path}")
    logging.debug(f"Sequence shape: {seq_data.shape}, dtype: {seq_data.dtype}")


def save_frame_data(seq_data: np.ndarray, output_dir: str, seq_id: str) -> None:
    """Save sequence data as individual frame files"""
    output_path = Path(output_dir) / seq_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_frames = seq_data.shape[0]
    logging.debug(f"Saving {num_frames} frames to {output_path}")
    
    for i in range(num_frames):
        frame_file = output_path / f"{i:04d}.npy"
        np.save(str(frame_file), seq_data[i])
        logging.debug(f"Saved frame {i+1}/{num_frames}: {frame_file}")
    
    logging.debug(f"Successfully saved {num_frames} frames to {output_path}")


def frames_to_sequence(input_dir: str, output_dir: Optional[str] = None) -> None:
    """Convert individual frame files to combined sequence files"""
    logging.info("Starting frames to sequence conversion...")
    
    if output_dir is None:
        output_dir = input_dir
    
    seq_dirs = find_sequence_dirs(input_dir)
    
    if not seq_dirs:
        logging.warning("No sequence directories found")
        return
    
    total_sequences = len(seq_dirs)
    successful_conversions = 0
    
    # Create progress bar with less frequent updates
    pbar = tqdm(seq_dirs, desc="Converting frames to sequences", unit="seq", 
                mininterval=0.5, maxinterval=2.0)
    
    start_time = time.time()
    
    for seq_dir in pbar:
        seq_path = Path(seq_dir)
        seq_id = seq_path.name
        
        # Update progress bar description less frequently
        if pbar.n % 10 == 0 or pbar.n == 0:
            pbar.set_description(f"Converting sequence {seq_id}")
        
        try:
            # Get frame files
            frame_files = get_frame_files(seq_dir)
            if not frame_files:
                logging.warning(f"No frame files found in {seq_dir}")
                continue
            
            # Load and combine frames
            seq_data = load_frame_data(frame_files)
            
            # Save combined sequence in the same directory as the frame folder
            output_file = Path(output_dir) / f"{seq_id}.npy"
            save_sequence_data(seq_data, str(output_file))
            
            # Remove the original frame directory
            import shutil
            shutil.rmtree(seq_dir)
            logging.debug(f"Removed frame directory: {seq_dir}")
            
            # Clear variables to free memory
            del seq_data
            del frame_files
            
            successful_conversions += 1
            logging.debug(f"Successfully converted sequence {seq_id}")
            
            # Periodic garbage collection to prevent memory buildup
            if pbar.n % 50 == 0:
                gc.collect()
            
        except Exception as e:
            logging.error(f"Failed to convert sequence {seq_id}: {e}")
            continue
    
    pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_seq = total_time / total_sequences if total_sequences > 0 else 0
    
    logging.info(f"Conversion completed: {successful_conversions}/{total_sequences} sequences converted successfully")
    logging.info(f"Total time: {total_time:.2f}s, Average time per sequence: {avg_time_per_seq:.2f}s")


def sequence_to_frames(input_dir: str, output_dir: Optional[str] = None) -> None:
    """Convert combined sequence files to individual frame files"""
    logging.info("Starting sequence to frames conversion...")
    
    if output_dir is None:
        output_dir = input_dir
    
    seq_files = find_sequence_files(input_dir)
    
    if not seq_files:
        logging.warning("No sequence files found")
        return
    
    total_sequences = len(seq_files)
    successful_conversions = 0
    
    # Create progress bar with less frequent updates
    pbar = tqdm(seq_files, desc="Converting sequences to frames", unit="seq",
                mininterval=0.5, maxinterval=2.0)
    
    start_time = time.time()
    
    for seq_file in pbar:
        seq_path = Path(seq_file)
        seq_id = seq_path.stem
        
        # Update progress bar description less frequently
        if pbar.n % 10 == 0 or pbar.n == 0:
            pbar.set_description(f"Converting sequence {seq_id}")
        
        try:
            # Load sequence data
            seq_data = np.load(seq_file)
            logging.debug(f"Loaded sequence {seq_id}, shape: {seq_data.shape}, dtype: {seq_data.dtype}")
            
            # Save as individual frames
            save_frame_data(seq_data, output_dir, seq_id)
            
            # Remove the original sequence file
            seq_path.unlink()
            logging.debug(f"Removed sequence file: {seq_file}")
            
            # Clear variables to free memory
            del seq_data
            
            successful_conversions += 1
            logging.debug(f"Successfully converted sequence {seq_id}")
            
            # Periodic garbage collection to prevent memory buildup
            if pbar.n % 50 == 0:
                gc.collect()
            
        except Exception as e:
            logging.error(f"Failed to convert sequence {seq_id}: {e}")
            continue
    
    pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_seq = total_time / total_sequences if total_sequences > 0 else 0
    
    logging.info(f"Conversion completed: {successful_conversions}/{total_sequences} sequences converted successfully")
    logging.info(f"Total time: {total_time:.2f}s, Average time per sequence: {avg_time_per_seq:.2f}s")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert between individual frame files and combined sequence files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert frames to sequences (in-place conversion)
  python convert_npy.py --mode frames_to_sequence --input_dir data/mmprism/collected_demo/mmwave
  
  # Convert sequences to frames (in-place conversion)
  python convert_npy.py --mode sequence_to_frames --input_dir data/mmprism/collected_demo/mmwave
  
  # Convert with custom output directory (preserves original files)
  python convert_npy.py --mode frames_to_sequence --input_dir data/mmprism/collected_demo/mmwave --output_dir data/mmprism/collected_demo/mmwave_converted
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["frames_to_sequence", "sequence_to_frames"],
        required=True,
        help="Conversion mode: frames_to_sequence or sequence_to_frames"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing the data to convert"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for converted data (default: same as input_dir, performs in-place conversion)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Perform conversion
    try:
        if args.mode == "frames_to_sequence":
            frames_to_sequence(args.input_dir, args.output_dir)
        elif args.mode == "sequence_to_frames":
            sequence_to_frames(args.input_dir, args.output_dir)
        else:
            logging.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
