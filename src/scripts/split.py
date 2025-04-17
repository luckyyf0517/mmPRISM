import os
import glob
import json
import numpy as np
import argparse

from tqdm import tqdm
from glob import glob
from termcolor import colored


def split_data(raw_data_root, seq_list, split_name, subfolder=''):
    for seq_id in seq_list:
        params = sorted(glob.glob(os.path.join(raw_data_root, seq_id, 'mmwave', '*.npy'))) # left one frame for differentation
        for frame_id, param in enumerate(params):
            assert '%06d' % frame_id in param, 'frame_id %06d not found in param: %s' % (frame_id, param)
            
            mmwave_path = os.path.join(raw_data_root, seq_id, 'mmwave', '%06d.npy' % frame_id)
            joints_path = os.path.join(raw_data_root, seq_id, 'joints', '%06d.npy' % frame_id)
            if os.path.exists(mmwave_path) & os.path.exists(joints_path): 
                data_dict['%06d' % len(data_dict)] = {
                    'seq_id': seq_id,
                    'frame_id': '%06d' % frame_id,
                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    signals_root = '/root/autodl-tmp/datasets/csl-news/signals'
    split_sub_folder = 'csl-news-demo01'
    signals_list = glob(os.path.join(signals_root, 'archive_*/*.npy'))
    print(colored(f'Total number of signals: {len(signals_list)}', 'green'))

    data_dict = {'all': {}, 'train': {}, 'val': {}}
    for index, signal_path in enumerate(tqdm(signals_list)):
        signal_name = os.path.basename(signal_path)
        seq_id = signal_name.split('.')[0]
        data_dict['all'][seq_id] = signal_path
        if index % 10 == 0:
            data_dict['val'][seq_id] = signal_path
        else:
            data_dict['train'][seq_id] = signal_path

    for split_name in ['train', 'val', 'all']:
        os.makedirs(os.path.join('dataset', split_sub_folder), exist_ok=True)
        with open(os.path.join('dataset', split_sub_folder, split_name + '.json'), 'w') as f:
            json.dump(data_dict[split_name], f, indent=2)
        print(colored(f'{split_name} data saved, total number of files: {len(data_dict[split_name])}', 'green'))
