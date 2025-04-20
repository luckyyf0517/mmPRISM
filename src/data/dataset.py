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
        with open(split_path, 'r') as f:
            self.data_dict = json.load(f)

        self.max_length = opt.get('max_length', 384)
        self.doppler_size = opt.get('doppler_size', 64)
        self.num_channels = opt.get('num_channels', 5)
        self.selected_channel = opt.get('selected_channel', 2)
            
        with open('dataset/CSL_News_Labels_converted.json', 'r') as f:
            self.caption_dict = json.load(f)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        id, signal_path = list(self.data_dict.items())[index]
        signal = np.load(signal_path) # (T, 64, C)
        signal = signal[..., self.selected_channel] # (T, 64)
        
        # Pad signal to fixed length of max_length frames
        T = signal.shape[0]
        pad_length = max(0, self.max_length - T)
        signal = np.pad(signal[:self.max_length], ((0, pad_length), (0, 0)), mode='constant')
        signal = np.log10(signal + 1e-6)
        signal = (signal - signal.min()) / (signal.max() - signal.min())
        signal = np.ascontiguousarray(signal)
        # Final shape should be (T, 64)
        assert signal.shape == (self.max_length, self.doppler_size), \
            f"Signal shape {signal.shape} does not match expected ({self.max_length}, {self.doppler_size})"
        
        caption = self.caption_dict[id]
        return {
            'id': id, 
            'signal': signal,
            'caption': caption,
        }
    

if __name__ == '__main__':
    opt = {
        'max_length': 384,
        'doppler_size': 64,
        'num_channels': 1,
        'selected_channel': 2
    }
    dataset = mmCSLNewsDataset(opt, split_path='dataset/csl-news-demo01/all.json')
    len_max = 0
    for item in tqdm(dataset):
        len_max = max(len_max, item['signal'].shape[0])
    print(len_max)