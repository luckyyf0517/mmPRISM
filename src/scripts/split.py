import os
import glob
import json
import numpy as np
import argparse
import hashlib

from tqdm import tqdm
from glob import glob
from termcolor import colored


def split_data(signals_list, subfolder='csl-news-demo01', val_ratio=0.1):
    """
    Split signal data into train and validation sets using deterministic hash-based splitting.
    This ensures that the same file will always be assigned to the same split regardless of execution order.
    
    Args:
        raw_data_root (str): Root directory containing the signal data
        signals_list (list): List of paths to signal files
        subfolder (str, optional): Subfolder name for saving split files. Defaults to 'csl-news-demo01'
        val_ratio (float, optional): Proportion of data to use for validation. Defaults to 0.1
    """
    # Initialize dictionaries to store split data
    data_dict = {'all': {}, 'train': {}, 'val': {}}
    
    for signal_path in signals_list:
        # Extract sequence ID from filename and generate deterministic hash
        seq_id = os.path.basename(signal_path).split('.')[0]
        hash_value = int(hashlib.md5(seq_id.encode()).hexdigest(), 16)
        # Determine split based on hash value (ensures consistent assignment)
        is_val = (hash_value % 100) < (val_ratio * 100)
        
        if os.path.exists(signal_path): 
            data_key = seq_id
            # Store file information
            data_dict['all'][data_key] = signal_path
            # Assign to appropriate split
            if is_val:
                data_dict['val'][data_key] = data_dict['all'][data_key]
            else:
                data_dict['train'][data_key] = data_dict['all'][data_key]
    
    # Save split information to JSON files
    for split_name in ['train', 'val', 'all']:
        os.makedirs(os.path.join('dataset', subfolder), exist_ok=True)
        with open(os.path.join('dataset', subfolder, split_name + '.json'), 'w') as f:
            json.dump(data_dict[split_name], f, indent=2)
        print(colored(f'{split_name} data saved, total number of files: {len(data_dict[split_name])}', 'green'))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals_root', type=str, default='/root/autodl-tmp/datasets/csl-news/poses/archive_0*/',
                       help='Root directory containing signal data')
    parser.add_argument('--subfolder', type=str, default='csl-news-demo01',
                       help='Subfolder name for saving split files')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                       help='Proportion of data to use for validation (between 0 and 1)')
    args = parser.parse_args()

    # Collect all signal files
    signals_list = sorted(glob(os.path.join(args.signals_root, '*.npy')))
    
    # signals_list_ = []
    # for signal_path in signals_list:
    #     if not os.path.exists(signal_path.replace('poses', 'pred_poses')):
    #         signals_list_.append(signal_path)
    # signals_list = signals_list_
    
    print(colored(f'Total number of signals: {len(signals_list)}', 'green'))
    
    # Perform data splitting
    split_data(signals_list, args.subfolder, args.val_ratio)
