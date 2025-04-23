from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional, Dict
from src.utils.tools import get_obj_from_str


class BaseDataInterface(LightningDataModule):
    """Base data interface for loading different datasets
    
    Args:
        cfg: Configuration object containing:
            - dataset: Dataset configuration including:
                - target: Path to dataset class
                - params: Parameters for dataset initialization
            - train_split: Path to training data split
            - val_split: Path to validation data split 
            - test_split: Path to test data split
            - num_workers: Number of workers for dataloaders
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.dataset_cfg = cfg.dataset
        self.train_split = cfg.train_split
        self.val_split = cfg.val_split
        self.test_split = cfg.test_split
        
        # Dataset class will be initialized in setup
        self.dataset_cls = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages
        
        Args:
            stage: 'fit', 'test', or None
        """
        # Import dataset class dynamically
        if self.dataset_cls is None:
            self.dataset_cls = get_obj_from_str(self.dataset_cfg)
        
        # Setup datasets based on stage
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_cls(
                split_path=self.train_split
            )
            self.val_dataset = self.dataset_cls(
                split_path=self.val_split
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_cls(
                split_path=self.test_split
            )
    
    def _get_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        """Get dataloader for dataset
        
        Args:
            dataset: Dataset instance
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,  # Fixed batch size of 1 for single image dataset
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, shuffle=False)