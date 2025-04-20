from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import mmCSLNewsDataset

class mmWaveDataInterface(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = mmCSLNewsDataset(
                self.cfg,
                split_path=self.cfg.train_split
            )
            self.val_dataset = mmCSLNewsDataset(
                self.cfg,
                split_path=self.cfg.val_split
            )
        
        if stage == 'test':
            self.test_dataset = mmCSLNewsDataset(
                self.cfg,
                split_path=self.cfg.test_split
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 