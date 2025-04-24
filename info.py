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
    
    print(f"\nTotal poses: {total_poses:,}")
    print(f"Total size: {format_size(total_size)}")
    

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
    

def analyze_features(base_path):
    """Analyze features folder content"""
    print(colored("\n=== Features Analysis ===", "blue"))
    feature_pattern = os.path.join(base_path, "features/archive_*/")
    feature_dirs = sorted(glob(feature_pattern))
    
    if not feature_dirs:
        print(colored("[X] No feature directories found", "red"))
        return
    
    total_features = 0
    total_size = 0
    feature_stats = defaultdict(int)
    
    print(f"Found {len(feature_dirs)} feature directories")
    for feature_dir in feature_dirs:
        archive_id = os.path.basename(os.path.dirname(feature_dir))
        features = glob(os.path.join(feature_dir, "**/*.npy"), recursive=True)
        
        dir_size = sum(os.path.getsize(f) for f in features)
        total_size += dir_size
        total_features += len(features)
        
        # Analyze categories
        for feature_path in features:
            category = feature_path.split('/')[-2]  # Get parent directory name
            feature_stats[category] += 1
    
    print(f"\nTotal features: {total_features:,}")
    print(f"Total size: {format_size(total_size)}")
    

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
    
    feature_dirs = glob(os.path.join(base_path, "features/archive_*/"))
    feature_ids = set(os.path.basename(os.path.dirname(f)).split('_')[1] for f in feature_dirs)
    
    total_archives = len(archive_ids)
    print(f"Total archives: {total_archives}")
    print(f"Processed poses: {len(pose_ids)} ({len(pose_ids)/total_archives*100:.1f}%)")
    print(f"Processed videos: {len(video_ids)} ({len(video_ids)/total_archives*100:.1f}%)")
    print(f"Processed features: {len(feature_ids)} ({len(feature_ids)/total_archives*100:.1f}%)")
    
    # Check processing pipeline consistency
    if pose_ids - feature_ids:
        print("\nPoses without features:")
        print("  " + ", ".join(sorted(pose_ids - feature_ids)))
    
    if archive_ids - pose_ids:
        print("\nUnprocessed archives (poses):")
        print("  " + ", ".join(sorted(archive_ids - pose_ids)))

def main():
    base_path = '/root/autodl-tmp/datasets/csl-news'
    print(colored(f"Analyzing dataset at: {base_path}", "cyan"))
    
    analyze_archives(base_path)
    analyze_poses(base_path)
    # analyze_videos(base_path)
    analyze_features(base_path)
    analyze_progress(base_path)

if __name__ == '__main__':
    main()