import os
import glob
import torch
import random
import argparse
from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mmwave2text.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='log/mmwave2text_v1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', '-n', dest='num_samples', type=int, default=3, help='Number of samples to inference')
    return parser.parse_args()

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

def main():
    args = parse_args()
    
    # Load configuration
    cfg = load_yaml(args.config)
    
    # Instantiate dataset from configuration
    data = instantiate_from_config(cfg.data_cfg)
    data.setup('fit')
    dataset = data.train_dataset
    
    # Load model
    latest_ckpt = get_latest_checkpoint(args.checkpoint_dir)
    print(f"Loading checkpoint from {latest_ckpt}")
    
    # Instantiate model from configuration
    model = instantiate_from_config(cfg.model_cfg)
    checkpoint = torch.load(latest_ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Generate multiple samples
    for i in range(args.num_samples):
        # Randomly select a sample
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        
        # Prepare input data
        signal = torch.from_numpy(sample['signal']).float().unsqueeze(0)
        signal = signal.to(args.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(signal, ["dummy text"])
            
            output_ids = model.generate(
                pre_compute_item=outputs,
                max_new_tokens=128,
                num_beams=5)
            generated_text = model.mt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Print results
        print(f"\nSample {i+1}/{args.num_samples}:")
        print("-" * 50)
        print(f"Sample ID: {sample['id']}")
        print(f"Ground Truth: {sample['caption']}")
        print(f"Generated: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    main()