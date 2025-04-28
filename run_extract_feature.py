import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from src.data.dataset import mmWaveSequenceDataset
from torch.utils.data import DataLoader
from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from easydict import EasyDict as edict
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

def main():
    # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print(colored("\n=== Feature Extraction Pipeline ===", "cyan", attrs=["bold"]))
        print(colored(f"Distributed setup:", "yellow"))
        print(colored(f"- World size: {world_size}", "yellow"))
        print(colored(f"- Backend: NCCL", "yellow"))
        print(colored(f"- GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.1f} GB", "yellow"))
    
    # Load OmniHand model from checkpoint
    args = edict({
        'config': 'config/omnihand.yaml',
        'resume_checkpoint': 'log/omnihand-0427/last.ckpt',
    })
    cfg = load_yaml(args.config)
    
    # Initialize model
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = 1
    model = instantiate_from_config(model_cfg)
    model = load_state_dict_from_zero_checkpoint(model, args.resume_checkpoint)
    model = model.to(device).eval()
    
    if local_rank == 0:
        print(colored("\nModel setup:", "yellow"))
        print(colored("- OmniHand model loaded and set to eval mode", "yellow"))
    
    # Initialize dataset with DistributedSampler
    dataset = mmWaveSequenceDataset(
        opt={'max_length': 512, 'mode': 'pose'},
        split_path='dataset/csl-news-demo02/all.json'
    )
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=4, 
        sampler=sampler
    )
    
    if local_rank == 0:
        print(colored("\nDataset setup:", "yellow"))
        print(colored(f"- Total samples: {len(dataset)}", "yellow"))
        print(colored(f"- Samples per GPU: {len(dataset) // world_size}", "yellow"))
        print(colored(f"- Batch size: 1", "yellow"))
        print(colored(f"- Num workers: 4", "yellow"))
    
    # Extract features
    dist.barrier()  # Synchronize all processes
    if local_rank == 0:
        print(colored("\nStarting feature extraction...", "green"))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, 
                         desc=colored(f"GPU {local_rank}", "blue"),
                         disable=local_rank != 0):
            # Get batch timestep data
            max_len = batch['valid_length'][0].item()
            points_t = batch['points_3d'][0, :max_len] # [T, 59, 3]
            velocities_t = batch['velocities_3d'][0, :max_len]  # [T, 59, 3]
            path = batch['path'][0]
                
            # Determine feature path
            feature_path = path.replace('poses', 'features')
            features = model.encode_feature(
                points_t.to(device), 
                velocities_t.to(device))  # [T, feature_dim]
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            np.save(feature_path, features.cpu().numpy())
            
            pose_path = path.replace('poses', 'pred_poses')
            joints_pred = model.forward_feature(features)
            joints_pred = joints_pred.cpu().numpy()
            os.makedirs(os.path.dirname(pose_path), exist_ok=True)
            np.save(pose_path, joints_pred)

    dist.barrier()  # Wait for all processes to complete
    if local_rank == 0:
        print(colored("\nFeature extraction completed!", "cyan", attrs=["bold"]))
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()