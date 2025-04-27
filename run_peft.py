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

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from src.utils.deepspeed_utils import add_deepspeed_args, get_train_ds_config


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Add local_rank argument
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher')
    
    # Original arguments
    parser.add_argument('--config', dest="config", default=None, required=False)
    parser.add_argument("--resume-checkpoint", default=None, type=str, required=False)
    parser.add_argument("--version", '-v', default=None, type=str, required=False)
    parser.add_argument('--seed', dest="seed", default=42, help="random seed")
    parser.add_argument('--test', dest="test", action="store_true", default=False)
    parser.add_argument('--reset', dest="reset", action="store_true", default=False)
    parser.add_argument('--batch-size', dest="batch_size", default=8, type=int)
    parser.add_argument('--gradient-accumulation-steps', dest="gradient_accumulation_steps", 
                       default=1, type=int)
    parser.add_argument('--max-epochs', dest="max_epochs", default=10, type=int)
    # DeepSpeed related arguments
    parser = add_deepspeed_args(parser)
    
    args = parser.parse_args()
    
    # Set distributed training related parameters
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        
    # Adjust batch size per GPU
    args.batch_size = args.batch_size // args.world_size
    
    return args


def main():
    args = parse_args()
    
    # Load configuration
    cfg = load_yaml(args.config)
    if args.version is None:
        args.version = args.config.replace('.yaml', '').replace('config/', '')
    
    # Initialize data interface
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = args.batch_size
    data = instantiate_from_config(data_cfg)
    
    # Create model
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = args.batch_size
    model = instantiate_from_config(model_cfg)
    
    # Set logger
    logger = WandbLogger(
        name=args.version, 
        project="mmwave-csl"
    )
    
    # if not args.resume_checkpoint and args.rank == 0: 
    #     log_file_list = glob.glob(os.path.join(cfg.log_dir, args.version, '*.ckpt'))
    #     if len(log_file_list) > 0:
    #         if args.reset: 
    #             shutil.rmtree(os.path.join(cfg.log_dir, args.version))
    #         else: 
    #             raise ValueError("Checkpoint files already exist. Please use --reset to restart a new experiment.")
    
    # Set callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.log_dir, args.version),
            filename="model-epoch-{epoch:02d}-val_loss-{valid/loss:.2f}",
            auto_insert_metric_name=False,
            save_top_k=3,
            monitor="valid/loss",
            mode="min",
            save_last=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Create DeepSpeed strategy
    strategy = DeepSpeedStrategy(
        stage=args.zero_stage,
        offload_optimizer=args.offload,
        offload_parameters=args.offload,
        offload_params_device="cpu" if args.offload else "none",
        precision_plugin=None,  # Let Trainer handle precision
        # ZeRO optimization configuration
        contiguous_gradients=True,
        overlap_comm=True,
        allgather_partitions=True,
        reduce_scatter=True,
        # Other configurations from get_train_ds_config
        config=get_train_ds_config(args)
    )

    precision_map = {
        'bf16': 'bf16-mixed',
        'fp16': 'fp16-mixed',
        'fp32': '32'
    }
    
    # Create trainer with temporary settings: 1 epoch, 10 steps
    trainer = Trainer(
        accelerator='gpu',
        devices=args.world_size,
        strategy=strategy,
        precision=precision_map[args.dtype],
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs, 
        num_sanity_val_steps=2,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        # limit_train_batches=10,  # Limit to 10 steps
        # limit_val_batches=10  # Limit to 10 steps
    )
    
    # Train model
    if not args.test:
        trainer.fit(model, datamodule=data, ckpt_path=args.resume_checkpoint)
    else:
        trainer.test(model, datamodule=data, ckpt_path=args.resume_checkpoint)
    

if __name__ == '__main__':
    main() 