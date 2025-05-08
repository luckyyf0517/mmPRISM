import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
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
        points_t = batch['points_3d'][0, :max_len]  # [T, 59, 3]
        velocities_t = batch['velocities_3d'][0, :max_len]  # [T, 59, 3]
        path = batch['path'][0]
        
        # Determine feature and pose paths
        feature_path = path.replace('poses', 'features')
        pose_path = path.replace('poses', 'pred_poses')
        
        # Check if both files already exist
        if os.path.exists(feature_path) and os.path.exists(pose_path):
            self.skipped_count += 1
            return
        
        # Process features in time batches
        features = self.process_time_batch(points_t, velocities_t)
        
        # Save features
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        np.save(feature_path, features.cpu().numpy())
        
        # Generate and save predicted poses
        joints_pred = self.forward_feature(features.to(self.device))
        joints_pred = joints_pred.cpu().numpy()
        os.makedirs(os.path.dirname(pose_path), exist_ok=True)
        np.save(pose_path, joints_pred)
        
        self.processed_count += 1
        
    def process_time_batch(self, points_t, velocities_t, time_batch_size=32):
        """Process a sequence in time batches"""
        # T = points_t.shape[0]
        # features_list = []
        
        # # Process in time batches
        # for t_start in range(0, T, time_batch_size):
        #     t_end = min(t_start + time_batch_size, T)
        #     points_batch = points_t[t_start:t_end].to(self.device)
        #     velocities_batch = velocities_t[t_start:t_end].to(self.device)
            
        #     # Get features for this time batch
        #     with torch.no_grad():
        #         features_batch = self.encode_feature(points_batch, velocities_batch)
        #     features_list.append(features_batch.cpu())
        
        # # Concatenate all time batches
        # return torch.cat(features_list, dim=0)
        
        return self.encode_feature(points_t, velocities_t)

def main():
    # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print(colored("\n=== Feature Extraction Pipeline ===", "cyan", attrs=["bold"]))
        print(colored(f"Distributed setup:", "yellow"))
        print(colored(f"- World size: {world_size}", "yellow"))
        print(colored(f"- GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB", "yellow"))
    
    # Load OmniHand model from checkpoint
    args = edict({
        'config': 'config/omnihand_base.yaml',
        'resume_checkpoint': 'log/omnihand/omnihand-0427/last.ckpt',
    })
    cfg = load_yaml(args.config)
    cfg.batch_size = 1
    
    # Modify test split to use all data
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.dataset = 'src.data.dataset.mmWaveSequenceDataset'
    data_cfg.params.cfg.batch_size = 1
    data_cfg.params.cfg.test_split = 'dataset/csl-news-to-extract/all.json'
    data_cfg.params.cfg.opt = {
        "caption_path": "dataset/CSL_News_Labels_converted.json",
        "max_length": 512,
        "modalities": {
            "use_features": False,
            "use_pred_pose": False,
            "use_raw_pose": True
        }
    }
    data = instantiate_from_config(data_cfg)
    data.setup('test')
    
    # Initialize model
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = 1
    model = FeatureExtractionOmniHand(model_cfg.params.cfg)
    
    # Create DeepSpeed strategy
    strategy = DeepSpeedStrategy(
        stage=2,
        offload_optimizer=False,
        offload_parameters=False,
        precision_plugin=None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        accelerator='gpu',
        devices=world_size,
        strategy=strategy,
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