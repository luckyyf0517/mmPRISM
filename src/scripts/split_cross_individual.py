"""
Cross-Individual Data Split Script for mmPRISM Dataset

This script splits mmPRISM radar data by sequence ID for cross-individual validation.

Data Organization:
- 0xxx sequences (e.g., 0000-0999) are located in: mmprism/collected_base/
- 1xxx sequences (e.g., 1000-1999) are located in: mmprism/collected_demo/

Split Configuration (Current Settings):
- Training set:   Sequences 0000-0099 (collected_base folder)
- Test/Val set:   Sequences 0100-0199 (collected_base folder)
- Output path:    dataset/collected-cross-individual/

Usage:
1. For collected_base data (0xxx sequences):
   python src/scripts/split_cross_individual.py

2. For collected_demo data (1xxx sequences):
   Modify the signals_root variable in the script

To modify split ranges, update the TRAIN_MIN, TRAIN_MAX, TEST_MIN, TEST_MAX variables
in the split_data_cross_individual() function.

"""

import os
import json
from glob import glob
from tqdm import tqdm
from termcolor import colored


def split_data_cross_individual(signals_list, subfolder='collected-cross-individual', dry_run=False):
    """
    Split signal data for cross-individual validation.
    - Training set: sequences 0000-0099 (inclusive)
    - Validation set: sequences 0100-0199 (inclusive)
    - Test set: same as validation set (0100-0199)

    Args:
        signals_list (list): List of paths to signal files
        subfolder (str, optional): Subfolder name for saving split files. Defaults to 'collected-cross-individual'
        dry_run (bool, optional): If True, only show split statistics without saving files. Defaults to False.
    """
    # Initialize dictionaries to store split data
    data_dict = {'all': {}, 'train': {}, 'test': {}, 'val': {}}
    split_stats = {'train': 0, 'test': 0, 'val': 0, 'all': 0}

    # Define sequence ID ranges
    TRAIN_MIN, TRAIN_MAX = 0, 99
    TEST_MIN, TEST_MAX = 100, 199

    print(colored(f'Processing {len(signals_list)} signal files...', 'yellow'))

    for signal_path in tqdm(signals_list, desc="Splitting data"):
        # Extract sequence ID from filename
        seq_id = os.path.basename(signal_path).split('.')[0]

        # Convert sequence ID to integer for range comparison
        try:
            seq_num = int(seq_id)
        except ValueError:
            # Skip files with non-numeric sequence IDs
            continue

        # Determine split based on sequence ID range
        split_name = None
        if TRAIN_MIN <= seq_num <= TRAIN_MAX:
            split_name = 'train'
        elif TEST_MIN <= seq_num <= TEST_MAX:
            split_name = 'val'  # Will be copied to both val and test

        if os.path.exists(signal_path) and split_name is not None:
            data_key = seq_id
            # Store file information
            data_dict['all'][data_key] = signal_path
            data_dict[split_name][data_key] = signal_path

            # Update statistics
            split_stats['all'] += 1
            split_stats[split_name] += 1

    # Copy validation data to test for cross-individual validation
    data_dict['test'] = data_dict['val'].copy()
    split_stats['test'] = split_stats['val']

    # Display split statistics
    print(colored('\n=== Split Statistics (Cross-Individual Validation) ===', 'cyan', attrs=['bold']))
    for split_name, count in split_stats.items():
        if count > 0:
            percentage = (count / split_stats['all']) * 100
            print(f'{split_name.upper():8}: {count:4} files ({percentage:5.1f}%)')

    # Show sample files for each split
    print(colored('\n=== Sample Files Preview ===', 'cyan', attrs=['bold']))
    for split_name in ['train', 'test', 'val']:
        if split_name in data_dict and data_dict[split_name]:
            sample_files = list(data_dict[split_name].keys())[:5]
            print(f'\n{split_name.upper()} samples (first 5):')
            for sample_id in sample_files:
                print(f'  - {sample_id}')

    # Save split information to JSON files (unless dry run)
    if not dry_run:
        print(colored(f'\nSaving split files to dataset/{subfolder}/...', 'green'))
        for split_name in ['train', 'test', 'val', 'all']:
            if data_dict[split_name]:  # Only save if split has data
                os.makedirs(os.path.join('dataset', subfolder), exist_ok=True)
                output_path = os.path.join('dataset', subfolder, split_name + '.json')
                with open(output_path, 'w') as f:
                    json.dump(data_dict[split_name], f, indent=2)
                print(colored(f'{split_name}.json saved with {len(data_dict[split_name])} files', 'green'))
    else:
        print(colored('\n=== DRY RUN MODE - No files were saved ===', 'yellow', attrs=['bold']))
        print(f'Would save to: dataset/{subfolder}/')
        print('To actually save the split files, run without --dry-run flag')


def analyze_individual_distribution(signals_list):
    """
    Analyze the distribution of files by sequence ID ranges

    Args:
        signals_list (list): List of paths to signal files
    """
    TRAIN_MIN, TRAIN_MAX = 0, 99
    TEST_MIN, TEST_MAX = 100, 199
    
    train_count = 0
    test_count = 0
    other_count = 0

    print(colored('=== Distribution Analysis (Sequence ID Ranges) ===', 'cyan', attrs=['bold']))

    for signal_path in signals_list:
        seq_id = os.path.basename(signal_path).split('.')[0]

        try:
            seq_num = int(seq_id)
            if TRAIN_MIN <= seq_num <= TRAIN_MAX:
                train_count += 1
            elif TEST_MIN <= seq_num <= TEST_MAX:
                test_count += 1
            else:
                other_count += 1
        except ValueError:
            other_count += 1

    total = train_count + test_count + other_count
    print(f'Total files: {total}')
    print(f'TRAIN range ({TRAIN_MIN:04d}-{TRAIN_MAX:04d}): {train_count:4} files ({train_count/total*100:5.1f}%)')
    print(f'TEST  range ({TEST_MIN:04d}-{TEST_MAX:04d}): {test_count:4} files ({test_count/total*100:5.1f}%)')
    print(f'Other sequences:                    {other_count:4} files ({other_count/total*100:5.1f}%)')

    # Show predicted split
    print(colored('\n=== Predicted Cross-Individual Split ===', 'cyan', attrs=['bold']))
    print(f'TRAIN ({TRAIN_MIN:04d}-{TRAIN_MAX:04d}): {train_count:4} files ({train_count/total*100:5.1f}%)')
    print(f'VAL   ({TEST_MIN:04d}-{TEST_MAX:04d}): {test_count:4} files ({test_count/total*100:5.1f}%)')
    print(f'TEST  ({TEST_MIN:04d}-{TEST_MAX:04d}): {test_count:4} files ({test_count/total*100:5.1f}%)')
    print(colored(f'\nNote: Sequences {TEST_MIN:04d}-{TEST_MAX:04d} are held out for cross-individual validation', 'yellow'))


if __name__ == '__main__':
    # Vobe coding: Hardcoded configuration
    signals_root = '/root/autodl-tmp/datasets/mmprism/collected_base/poses/'
    subfolder = 'collected-cross-individual'
    pattern = '*.npy'
    analyze_only = False  # Set to True to only analyze without saving
    dry_run = False       # Set to True to preview split without saving files

    # Collect all signal files
    signals_list = sorted(glob(os.path.join(signals_root, pattern)))

    if not signals_list:
        print(colored(f'No files found in {signals_root} with pattern {pattern}', 'red'))
        exit(1)

    print(colored(f'Found {len(signals_list)} files matching pattern: {pattern}', 'green'))

    # Analyze individual distribution
    analyze_individual_distribution(signals_list)

    # Perform splitting unless analyze_only
    if not analyze_only:
        print(colored(f'\n=== Performing Cross-Individual Split ===', 'cyan', attrs=['bold']))
        print(f'Split rule: Sequences 0000-0099 -> TRAIN, Sequences 0100-0199 -> VAL & TEST')
        split_data_cross_individual(signals_list, subfolder, dry_run)
    else:
        print(colored('\n=== Analysis Complete ===', 'yellow', attrs=['bold']))