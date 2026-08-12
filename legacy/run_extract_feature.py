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

class FeatureExtractionOmniHand(OmniHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_count = 0
        self.processed_count = 0
        
    def test_step(self, batch, batch_idx):
        # Get batch timestep data
        max_len = batch['valid_length'][0].item()
        points_t = batch['joints'][0, :max_len]  # [T, 59, 3]
        path = batch['path'][0]
        
        # Determine feature and pose paths
        pose_path = path.replace('poses', 'pred_poses')
        
        # # Check if both files already exist
        # if os.path.exists(pose_path):
        #     self.skipped_count += 1
        #     return
        
        # Extract features using encode_feature method
        if self.use_simulator:
            velocities_t = batch['velocities'][0, :max_len]  # [T, 59, 3]
            input_data = {
                'joints': points_t,
                'velocities': velocities_t
            }
        else:
            input_data = {'mmwave': batch['mmwave'][0, :max_len]}
            
        features = self.encode_feature(input_data)

        # # Save features
        # feature_path = path.replace('poses', 'features_0602_rtm')
        # os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        # np.save(feature_path, features.cpu().numpy())
        
        # Generate and save predicted poses
        joints_pred = self.forward_feature(features.to(self.device))[..., :3]
        
        # Save predicted poses
        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        np.save(pose_path, joints_pred.cpu().numpy())
        
        self.processed_count += 1

        # print MPJPE
        valid_mask = ~torch.any(torch.isnan(points_t), dim=-1)
        pred_valid = joints_pred[valid_mask]
        target_valid = points_t[valid_mask]
        mpjpe = torch.norm(pred_valid - target_valid, dim=-1).mean() * 1e3
        self.print(f"MPJPE: {mpjpe.mean()}")
        
def main():
    # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    datastage = 'daily'
    
    if local_rank == 0:
        print(colored("\n=== Feature Extraction Pipeline ===", "cyan", attrs=["bold"]))
        print(colored(f"Distributed setup:", "yellow"))
        print(colored(f"- World size: {world_size}", "yellow"))
        print(colored(f"- GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB", "yellow"))
    
    # Modify test split to use all data
    assert datastage == 'daily'
    args = edict({
        'config': 'config/omnihand/omnihand_rtm_collected.yaml',
        'resume_checkpoint': 'log/omnihand/omnihand-rtm-collected-0704-demo/last.ckpt',
    })
    cfg = load_yaml(args.config)
    cfg.batch_size = 1
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.dataset = 'src.data.dataset.CollectedDailyDataset'
    data_cfg.params.cfg.batch_size = 1
    data_cfg.params.cfg.test_split = 'dataset/collected-demo/all.json'
    
    data_cfg.params.cfg.opt = {
        "annotation_path": 'data/csl-daily/sentence_label/csl2020ct_v2.pkl',
        "max_length": 512,
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
    model = FeatureExtractionOmniHand(model_cfg.params.cfg)
    
    # Initialize trainer with DDP strategy
    trainer = Trainer(
        accelerator='gpu',
        devices=world_size,
        strategy='ddp',
        precision='32',
        enable_progress_bar=True,
    )
    
    # Run feature extraction
    if local_rank == 0:
        print(colored("\nStarting feature extraction...", "green"))
    
    trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path=args.resume_checkpoint)
    
    if local_rank == 0:
        print(colored("\nFeature extraction completed!", "cyan", attrs=["bold"]))
        print(colored(f"- Skipped: {model.skipped_count} files", "yellow"))
        print(colored(f"- Processed: {model.processed_count} files", "yellow"))

if __name__ == "__main__":
    main()