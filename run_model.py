import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*Checkpoint directory*")
warnings.filterwarnings('ignore', '.*find_unused_parameters=True was specified.*')
warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
warnings.filterwarnings('ignore', '.*0NCCL_AVOID_RECORD_STREAMS=1 has no effect for point-to-point collectives.*')

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest="config", default=None, required=False)
    parser.add_argument("--resume-checkpoint", default=None, type=str, required=False)
    parser.add_argument("--version", '-v', default=None, type=str, required=False)
    parser.add_argument('--seed', dest="seed", default=42, help="random seed")
    parser.add_argument('--test', dest="test", action="store_true", default=False)
    parser.add_argument('--reset', dest="reset", action="store_true", default=False)
    
    args = parser.parse_args()
    if args.test: 
        assert args.resume_checkpoint is not None
    
    args.rank = int(os.environ.get('RANK')) if 'RANK' in os.environ else 0
    args.world_size = int(os.environ.get('WORLD_SIZE')) if 'WORLD_SIZE' in os.environ else 1
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # Load configuration
    cfg = load_yaml(args.config)
    if args.version is None:
        args.version = args.config.replace('.yaml', '').replace('config/', '')
    
    # Create log directory
    os.makedirs(os.path.join(cfg.log_dir, args.version), exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.log_dir, args.version, 'config.yaml'))
    
    # Set random seed
    set_seed(args.seed, torch.cuda.device_count())
    
    # Initialize model and data interface from configuration
    model = instantiate_from_config(cfg.model_cfg)
    data = instantiate_from_config(cfg.data_cfg)
    
    # Set logger
    logger = WandbLogger(name=args.version, project='mmWave2Text')
    if not args.resume_checkpoint and args.rank == 0: 
        log_file_list = glob.glob(os.path.join(cfg.log_dir, args.version, '*.ckpt'))
        if len(log_file_list) > 0:
            if args.reset: 
                for log_file in log_file_list: 
                    os.remove(log_file) 
            else: 
                raise ValueError("Checkpoint files already exist. Please use --reset to restart a new experiment.")
        
    # Set checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, args.version),
        monitor='valid/loss',
        filename='epoch_{epoch:02d}_val_{valid/loss:.4f}',
        save_top_k=10,
        mode='min',
        auto_insert_metric_name=False, 
        save_last=True,
        save_weights_only=False,)
    
    # Initialize trainer
    trainer = Trainer(
        accelerator='gpu', 
        devices=args.world_size, 
        strategy=cfg.strategy, 
        logger=logger,
        log_every_n_steps=1,
        max_epochs=cfg.max_epochs, 
        num_sanity_val_steps=2, # run validation step experimentaly
        reload_dataloaders_every_n_epochs=1, 
        callbacks=[checkpoint_callback], 
        enable_progress_bar=True)

    if not args.test:
        trainer.fit(model, datamodule=data, ckpt_path=args.resume_checkpoint)
    else:
        trainer.test(model, datamodule=data, ckpt_path=args.resume_checkpoint)