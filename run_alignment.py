import os
import sys
sys.path.append('.')

import torch
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from src.utils.deepspeed_utils import add_deepspeed_args, get_train_ds_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/alignment.yaml')
    parser.add_argument('--version', default='alignment_v1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--local_rank', type=int, default=-1)
    
    # Add DeepSpeed arguments
    parser = add_deepspeed_args(parser)
    args = parser.parse_args()
    
    # Set distributed training parameters
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    args.batch_size = args.batch_size // args.world_size
    return args

def main():
    args = parse_args()
    
    # Load config
    cfg = load_yaml(args.config)
    cfg.data_cfg.params.cfg.batch_size = args.batch_size
    cfg.model_cfg.params.cfg.training.batch_size = args.batch_size
    
    # Create model and data interface
    model = instantiate_from_config(cfg.model_cfg)
    data = instantiate_from_config(cfg.data_cfg)
    
    # Setup logger
    logger = WandbLogger(name=args.version, project='omniHand_alignment')
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.log_dir, args.version),
        filename='epoch_{epoch:02d}_loss_{valid/loss:.4f}',
        monitor='valid/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    # Setup DeepSpeed strategy
    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=args.offload,
        offload_parameters=args.offload,
        contiguous_gradients=True,
        overlap_comm=True,
        config=get_train_ds_config(args)
    )
    
    # Create trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=args.world_size,
        strategy=strategy,
        precision='bf16-mixed',
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.max_epochs,
        log_every_n_steps=1
    )
    
    # Start training
    data.setup('fit')
    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    main()