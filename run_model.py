import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["WANDB_MODE"] = "offline"

import yaml
import glob
import wandb
import shutil
import random
import argparse
import logging
import numpy as np

import sys
sys.path.append('.')

import torch
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config

def set_seed(seed, n_gpu):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # Basic training arguments
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--config', dest="config", default=None, required=False)
    parser.add_argument("--resume-checkpoint", default=None, type=str)
    parser.add_argument("--version", '-v', default=None, type=str)
    parser.add_argument('--seed', dest="seed", default=42)
    parser.add_argument('--test', dest="test", action="store_true", default=False)
    parser.add_argument('--reset', dest="reset", action="store_true", default=False)
    parser.add_argument('--batch-size', dest="batch_size", default=32, type=int)
    parser.add_argument('--max-epochs', dest="max_epochs", default=100, type=int)
    parser.add_argument('--precision', choices=['32', '16', 'bf16'], default='32',
                       help='Training precision: 32 for full precision, 16 for mixed precision, bf16 for bfloat16')
    
    args = parser.parse_args()
    
    # Handle distributed training setup
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        
    # Scale batch size by number of GPUs
    args.batch_size = args.batch_size // args.world_size
    
    return args

def main():
    args = parse_args()
    
    # Load and process configuration
    cfg = load_yaml(args.config)
    if args.version is None:
        args.version = args.config.replace('.yaml', '').replace('config/', '')
    cfg.data_cfg.params.cfg.batch_size = args.batch_size
    cfg.model_cfg.params.cfg.batch_size = args.batch_size
    
    # Setup logging directory
    os.makedirs(os.path.join(cfg.log_dir, args.version), exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.log_dir, args.version, 'config.yaml'))
    
    # Set random seed for reproducibility
    set_seed(args.seed, args.world_size)
    
    # Initialize model and data module
    model = instantiate_from_config(cfg.model_cfg)
    data = instantiate_from_config(cfg.data_cfg)
    
    # Setup wandb logger
    logger = WandbLogger(name=args.version, project='omniHand')
    if not args.resume_checkpoint and args.rank == 0: 
        log_file_list = glob.glob(os.path.join(cfg.log_dir, args.version, '*.ckpt'))
        if len(log_file_list) > 0:
            if args.reset: 
                shutil.rmtree(os.path.join(cfg.log_dir, args.version))
            else: 
                raise ValueError("Checkpoint files already exist. Use --reset to restart.")
        
    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, args.version),
        monitor='epoch',
        filename='epoch_{epoch:02d}_MPJPE_{valid/MPJPE:.4f}',
        save_top_k=5,
        mode='max',
        auto_insert_metric_name=False, 
        save_last=True,
        save_weights_only=False,
        enable_version_counter=False)

    # Setup DDP strategy
    strategy = DDPStrategy(
        find_unused_parameters=True,  # Usually not needed
        process_group_backend="nccl"   # NCCL backend for GPU training
    )
    
    # Configure trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=args.world_size,
        strategy=strategy,
        precision=args.precision,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=2,
        reload_dataloaders_every_n_epochs=1, 
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        # limit_train_batches=10,  
        # limit_val_batches=10, 
        # limit_test_batches=10 
    )

    # Run training or testing
    if not args.test:
        data.setup('fit')
        trainer.fit(model, datamodule=data, ckpt_path=args.resume_checkpoint)
    else:
        data.setup('test')
        trainer.test(model, datamodule=data, ckpt_path=args.resume_checkpoint)

if __name__ == '__main__':
    main()