import os
import glob
import torch
import random
import argparse
from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/mmwave2text.yaml')
    parser.add_argument('--checkpoint_dir', type=str, default='log/deepspeed_training')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_samples', '-n', dest='num_samples', type=int, default=3, help='Number of samples to inference')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    cfg = load_yaml(args.config)
    
    # Instantiate dataset from configuration
    cfg.data_cfg.params.cfg.batch_size = 1
    data = instantiate_from_config(cfg.data_cfg)
    data.setup('fit')
    dataset = data.train_dataset
    
    # Instantiate model from configuration
    cfg.model_cfg.params.cfg.batch_size = 1
    model = instantiate_from_config(cfg.model_cfg)
    
    # Ensure the checkpoint path is correct (includes last.ckpt directory)
    if not args.checkpoint_dir.endswith('last.ckpt'):
        checkpoint_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
    else:
        checkpoint_dir = args.checkpoint_dir
        
    # Load checkpoint using the function provided by DeepSpeed
    load_state_dict_from_zero_checkpoint(model, checkpoint_dir)
    
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