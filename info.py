import os
from glob import glob
from collections import defaultdict
import numpy as np
from termcolor import colored
import zipfile

def format_size(size_bytes):
    """Convert size in bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def analyze_archives(base_path):
    """Analyze archives folder content"""
    print(colored("\n=== Archives Analysis ===", "blue"))
    archive_pattern = os.path.join(base_path, "archives/archive_*.zip")
    archives = sorted(glob(archive_pattern))
    
    if not archives:
        print(colored("[X] No archives found", "red"))
        return
    
    total_size = 0
    archive_stats = defaultdict(int)
    
    print(f"Found {len(archives)} archives")
    for archive_path in archives:
        size = os.path.getsize(archive_path)
        total_size += size
        
        # Count files in zip
        with zipfile.ZipFile(archive_path, 'r') as zf:
            files = zf.namelist()
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext:
                    archive_stats[ext] += 1
    
    print(f"\nTotal size: {format_size(total_size)}")
    print("\nFile types distribution:")
    for ext, count in sorted(archive_stats.items()):
        print(f"  {ext}: {count:,} files")

def analyze_poses(base_path):
    """Analyze poses folder content"""
    print(colored("\n=== Poses Analysis ===", "blue"))
    pose_pattern = os.path.join(base_path, "poses/archive_*/")
    pose_dirs = sorted(glob(pose_pattern))
    
    if not pose_dirs:
        print(colored("[X] No pose directories found", "red"))
        return
    
    total_poses = 0
    total_size = 0
    pose_stats = defaultdict(int)
    frame_stats = []
    
    print(f"Found {len(pose_dirs)} pose directories")
    for pose_dir in pose_dirs:
        archive_id = os.path.basename(os.path.dirname(pose_dir))
        poses = glob(os.path.join(pose_dir, "**/*.npy"), recursive=True)
        
        dir_size = sum(os.path.getsize(f) for f in poses)
        total_size += dir_size
        total_poses += len(poses)
        
        # Analyze categories
        for pose_path in poses:
            category = pose_path.split('/')[-2]  # Get parent directory name
            pose_stats[category] += 1
            
            # Analyze frame counts
            try:
                data = np.load(pose_path)
                frame_stats.append(len(data))
            except:
                continue
    
    print(f"\nTotal poses: {total_poses:,}")
    print(f"Total size: {format_size(total_size)}")
    
    if frame_stats:
        print(f"\nFrame statistics:")
        print(f"  Average frames: {np.mean(frame_stats):.1f}")
        print(f"  Median frames: {np.median(frame_stats):.1f}")
        print(f"  Min frames: {np.min(frame_stats)}")
        print(f"  Max frames: {np.max(frame_stats)}")
    
    print("\nCategory distribution:")
    for category, count in sorted(pose_stats.items()):
        print(f"  {category}: {count:,} files")

def analyze_videos(base_path):
    """Analyze videos folder content"""
    print(colored("\n=== Videos Analysis ===", "blue"))
    video_pattern = os.path.join(base_path, "videos/archive_*/")
    video_dirs = sorted(glob(video_pattern))
    
    if not video_dirs:
        print(colored("[X] No video directories found", "red"))
        return
    
    total_videos = 0
    total_size = 0
    video_stats = defaultdict(int)
    
    print(f"Found {len(video_dirs)} video directories")
    for video_dir in video_dirs:
        if not os.path.exists(video_dir):  # Skip if directory doesn't exist
            continue
            
        videos = glob(os.path.join(video_dir, "**/*.mp4"), recursive=True)
        dir_size = sum(os.path.getsize(f) for f in videos)
        total_size += dir_size
        total_videos += len(videos)
        
        # Analyze categories
        for video_path in videos:
            category = video_path.split('/')[-2]
            video_stats[category] += 1
    
    print(f"\nTotal videos: {total_videos:,}")
    print(f"Total size: {format_size(total_size)}")
    
    print("\nCategory distribution:")
    for category, count in sorted(video_stats.items()):
        print(f"  {category}: {count:,} files")

def analyze_signals(base_path):
    """Analyze signals folder content"""
    print(colored("\n=== Signals Analysis ===", "blue"))
    signal_pattern = os.path.join(base_path, "signals/archive_*/")
    signal_dirs = sorted(glob(signal_pattern))
    
    if not signal_dirs:
        print(colored("[X] No signal directories found", "red"))
        return
    
    total_signals = 0
    total_size = 0
    signal_stats = defaultdict(int)
    signal_dims = []
    
    print(f"Found {len(signal_dirs)} signal directories")
    for signal_dir in signal_dirs:
        archive_id = os.path.basename(os.path.dirname(signal_dir))
        signals = glob(os.path.join(signal_dir, "**/*.npy"), recursive=True)
        
        dir_size = sum(os.path.getsize(f) for f in signals)
        total_size += dir_size
        total_signals += len(signals)
        
        # Analyze categories and dimensions
        for signal_path in signals:
            category = signal_path.split('/')[-2]  # Get parent directory name
            signal_stats[category] += 1
            
            # Analyze signal dimensions
            try:
                data = np.load(signal_path)
                signal_dims.append(data.shape)  # [frames, 64, 3] for doppler, azimuth, elevation
            except:
                continue
    
    print(f"\nTotal signals: {total_signals:,}")
    print(f"Total size: {format_size(total_size)}")
    
    if signal_dims:
        frame_lengths = [dim[0] for dim in signal_dims]
        print(f"\nSignal statistics:")
        print(f"  Signal shape: {signal_dims[0]}")  # Should be consistent
        print(f"  Average frames: {np.mean(frame_lengths):.1f}")
        print(f"  Median frames: {np.median(frame_lengths):.1f}")
        print(f"  Min frames: {np.min(frame_lengths)}")
        print(f"  Max frames: {np.max(frame_lengths)}")
    
    print("\nCategory distribution:")
    for category, count in sorted(signal_stats.items()):
        print(f"  {category}: {count:,} files")

def analyze_progress(base_path):
    """Analyze processing progress"""
    print(colored("\n=== Processing Progress ===", "blue"))
    
    # Get all archives
    archives = glob(os.path.join(base_path, "archives/archive_*.zip"))
    archive_ids = set(os.path.basename(a).split('_')[1].split('.')[0] for a in archives)
    
    # Get processed data
    pose_dirs = glob(os.path.join(base_path, "poses/archive_*/"))
    pose_ids = set(os.path.basename(os.path.dirname(p)).split('_')[1] for p in pose_dirs)
    
    video_dirs = glob(os.path.join(base_path, "videos/archive_*/"))
    video_ids = set(os.path.basename(os.path.dirname(v)).split('_')[1] for v in video_dirs)
    
    signal_dirs = glob(os.path.join(base_path, "signals/archive_*/"))
    signal_ids = set(os.path.basename(os.path.dirname(s)).split('_')[1] for s in signal_dirs)
    
    total_archives = len(archive_ids)
    print(f"Total archives: {total_archives}")
    print(f"Processed poses: {len(pose_ids)} ({len(pose_ids)/total_archives*100:.1f}%)")
    print(f"Processed videos: {len(video_ids)} ({len(video_ids)/total_archives*100:.1f}%)")
    print(f"Processed signals: {len(signal_ids)} ({len(signal_ids)/total_archives*100:.1f}%)")
    
    # Check processing pipeline consistency
    if pose_ids - signal_ids:
        print("\nPoses without signals:")
        print("  " + ", ".join(sorted(pose_ids - signal_ids)))
    
    if archive_ids - pose_ids:
        print("\nUnprocessed archives (poses):")
        print("  " + ", ".join(sorted(archive_ids - pose_ids)))

def main():
    base_path = '/root/autodl-tmp/datasets/csl-news'
    print(colored(f"Analyzing dataset at: {base_path}", "cyan"))
    
    analyze_archives(base_path)
    # analyze_poses(base_path)
    # analyze_videos(base_path)
    analyze_signals(base_path)
    analyze_progress(base_path)

if __name__ == '__main__':
    main()