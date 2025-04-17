import os
import cv2
import glob
import time
import json
import torch
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class mmCSLNewsDataset(Dataset):
    def __init__(self, opt, split_path=None):
        self.opt = opt
        self.data_root = opt.get('data_root', None)
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)
            
        with open('dataset/CSL_News_Labels_converted.json', 'r') as f:
            self.caption_dict = json.load(f)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        id, signal_path = list(self.data_dict.items())[index]
        signal = np.load(signal_path) # (T, 64, 3)
        
        # Pad signal to fixed length of 512 frames
        T = signal.shape[0]
        if T > 512:
            signal = signal[:512]
        elif T < 512:
            pad_length = 512 - T
            # Pad with zeros along time dimension
            signal = np.pad(signal, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
        # Final shape should be (512, 64, 3)
        assert signal.shape == (512, 64, 3), f"Signal shape {signal.shape} does not match expected (512, 64, 3)"
        
        caption = self.caption_dict[id]
        return {
            'id': id, 
            'signal': signal,
            'caption': caption,
        }
    

if __name__ == '__main__':
    opt = {
        'data_root': '/root/autodl-tmp/datasets/csl-news/signals',
    }
    dataset = mmCSLNewsDataset(opt, split_path='dataset/csl-news-demo01/all.json')
    len_max = 0
    for item in tqdm(dataset):
        len_max = max(len_max, item['signal'].shape[0])
    print(len_max)