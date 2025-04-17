import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Sampler

from src.data.dataset import mmCSLNewsDataset


class DInterface(pl.LightningDataModule):
    def __init__(self, cfg):
        super(DInterface, self).__init__()
        self.cfg = cfg
        self.sample_ratio = cfg.sample_ratio
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
    
    def train_dataloader(self):
        dataset = self.train_dataset
        sampler = SubsetSampler([len(dataset)], shuffle=True, sample_ratio=self.sample_ratio)
        return DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False, 
            drop_last=True)

    def val_dataloader(self):
        dataset = self.val_dataset
        sampler = SubsetSampler([len(dataset)], shuffle=False, sample_ratio=self.sample_ratio)
        return DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False, 
            drop_last=True)
    
    def test_dataloader(self):
        dataset = self.test_dataset
        sampler = SubsetSampler([len(dataset)], shuffle=False, sample_ratio=self.sample_ratio)
        return DataLoader(
            dataset, 
            sampler=sampler,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False, 
            drop_last=True)


class SubsetSampler(Sampler):
    def __init__(self, dataset_sizes, shuffle=False, sample_ratio=1):
        self.dataset_sizes = dataset_sizes
        self.indices = []
        current_indice = 0
        for dataset_size in dataset_sizes:
            if shuffle:
                self.indices.extend(torch.randperm(dataset_size) + current_indice)
            else:
                self.indices.extend(torch.arange(dataset_size) + current_indice)
            current_indice += dataset_size
        self.indices = self.indices[::sample_ratio]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class mmCSLNewsDInterface(DInterface):
    def __init__(self, cfg):
        super(mmCSLNewsDInterface, self).__init__(cfg)

    def setup(self, stage=None):
        cfg = self.cfg
        if stage == 'fit': 
            self.train_dataset = mmCSLNewsDataset(cfg.opt, cfg.train_split) 
            self.val_dataset = mmCSLNewsDataset(cfg.opt, cfg.val_split) 
        elif stage == 'test': 
            self.test_dataset = mmCSLNewsDataset(cfg.opt, cfg.test_split) 
        else: 
            raise NotImplementedError

