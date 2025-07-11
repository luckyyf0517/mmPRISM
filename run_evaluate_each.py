import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from pytorch_lightning import Trainer
from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from easydict import EasyDict as edict
from src.model.omnihand import OmniHand

class MPJPEEvaluator(OmniHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_count = 0
        self.processed_count = 0
        
    def test_step(self, batch, batch_idx):
        # Get batch timestep data
        max_len = batch['valid_length'][0].item()
        points_t = batch['joints'][0, :max_len]  # [T, 2, 24, 3]
        mmwave_t = batch['mmwave'][0, :max_len]  # [T, 64, 32, 32]

        # Process doppler data
        mmwave = self.processor(mmwave_t)
        features = self.backbone(mmwave)

        # Generate predicted poses
        joints_pred = self.forward_feature(features.to(self.device))[..., :3]
        
        # Calculate MPJPE
        valid_mask = ~torch.any(torch.isnan(points_t), dim=-1)
        pred_valid = joints_pred[valid_mask]
        target_valid = points_t[valid_mask]
        mpjpe = torch.norm(pred_valid - target_valid, dim=-1).mean() * 1e3
        print(f"ID: {batch['id'][0]}, MPJPE: {mpjpe.item()}")
        
        self.processed_count += 1

def main():
    # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    datastage = 'daily'
    
    if local_rank == 0:
        print(colored("\n=== MPJPE Evaluation Pipeline ===", "cyan", attrs=["bold"]))
        print(colored(f"Distributed setup:", "yellow"))
        print(colored(f"- World size: {world_size}", "yellow"))
        print(colored(f"- GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB", "yellow"))
    
    # Modify test split to use all data
    assert datastage == 'daily'
    args = edict({
        # 'config': 'config/omnihand_rtm_collected.yaml',
        # 'resume_checkpoint': 'log/omnihand/omnihand-rtm-collected-0704/last.ckpt',
        'config': 'config/omnihand_mmhand_collected.yaml',
        'resume_checkpoint': 'log/omnihand/omnihand-mmhand-collected-0707/last.ckpt',
    })
    cfg = load_yaml(args.config)
    cfg.batch_size = 1
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.dataset = 'src.data.dataset.CollectedDailyDataset'
    data_cfg.params.cfg.batch_size = 1
    data_cfg.params.cfg.test_split = 'dataset/collected-500/val.json'
    data_cfg.params.cfg.opt = {
        "annotation_path": None,
        "max_length": 100, 
        "modalities": {
            "use_pred_pose": True,
            "use_raw_pose": False, 
            "use_gt_pose": False, 
            "use_features": False,
            "use_mmwave": True,
        },
        "pose_config": {
            "pose_dir": "poses", 
            "norm_pose": True,
        },
    }
        
    data = instantiate_from_config(data_cfg)
    data.setup('test')
    
    # Initialize model
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = 1
    model = MPJPEEvaluator(model_cfg.params.cfg)
    
    # Initialize trainer with DDP strategy
    trainer = Trainer(
        accelerator='gpu',
        devices=world_size,
        strategy='ddp',
        precision='32',
        enable_progress_bar=False,
    )
    
    # Run MPJPE evaluation
    if local_rank == 0:
        print(colored("\nStarting MPJPE evaluation...", "green"))
    
    trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path=args.resume_checkpoint)
    
    if local_rank == 0:
        print(colored("\nMPJPE evaluation completed!", "cyan", attrs=["bold"]))
        print(colored(f"- Skipped: {model.skipped_count} files", "yellow"))
        print(colored(f"- Processed: {model.processed_count} files", "yellow"))

if __name__ == "__main__":
    main()