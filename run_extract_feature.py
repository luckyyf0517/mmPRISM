import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from src.data.dataset import mmWaveSequenceDataset
from src.model.cubenet import CubeNet
from torch.utils.data import DataLoader
from src.fmcw.simulator import Simulation
from src.utils.tools import instantiate_from_config
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


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
    
    # Initialize models (no DDP wrapper needed)
    simulator = Simulation().to(device).eval()
    
    cubenet_cfg = {
        'target': 'src.model.cubenet.CubeNet',
        'params': {
            'input_dim': 32,
            'hidden_dims': [64, 128, 256, 512],
            'num_blocks': [3, 4, 6, 3], 
            'strides': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 1]],
            'block': 'src.model.cubenet.BasicBlock3D',
            'norm_layer': 'torch.nn.GroupNorm'
        },
    }
    backbone = instantiate_from_config(cubenet_cfg)
    backbone.load_state_dict(torch.load(
        os.path.join('weights/cubenet/model.pth'), weights_only=True), strict=True)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval().to(device)
    
    if local_rank == 0:
        print(colored("\nModel setup:", "yellow"))
        print(colored("- Models moved to GPU and set to eval mode", "yellow"))
    
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
            # Get batch b timestep data
            max_len = batch['valid_length'][0].item()
            points_t = batch['points_3d'][0, :max_len]      # [T, 59, 3]
            velocities_t = batch['velocities_3d'][0, :max_len]  # [T, 59, 3]
            path = batch['path'][0]
                
            # Determine feature path
            feature_path = path.replace('poses', 'features')
            if os.path.exists(feature_path):
                continue
            
            # Simulate and extract features
            mmwave = simulator(points_t.to(device), velocities_t.to(device))  # [T, 32, 32, 32, 32]
            features = backbone(mmwave)  # [T, feature_dim]
            features = torch.nn.functional.adaptive_max_pool3d(features, (1, 1, 1))
            features = features.squeeze(-1).squeeze(-1).squeeze(-1)  # [T, feature_dim]

            # Save features for each sample in batch
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            np.save(feature_path, features.cpu().numpy())
            
            # # Print save info for all ranks
            # print(colored(f"[Rank {local_rank}] Saved to {feature_path}", "green"))

    dist.barrier()  # Wait for all processes to complete
    if local_rank == 0:
        print(colored("\nFeature extraction completed!", "cyan", attrs=["bold"]))
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()