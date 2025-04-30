import os
os.environ['TOKENIZERS_PARALLELISM'] = 'TRUE'

import random
import argparse
import torch
from tqdm import tqdm
from termcolor import colored
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import sys
sys.path.append('.')

from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from src.model.llm.text_processor import insert_wave_tokens, create_conversation, prepare_conversation_data, prepare_simple_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_batch(samples, model, tokenizer, conv_type='role'):
    # Stack features and collect captions/valid_lengths
    # poses = torch.stack([s['joints'] for s in samples], dim=0)  # [B, N, 2, 24, 3]
    # wave_embeds = model.hand_pose_encoder(poses.cuda())  # [B, N, C]
    wave_embeds = torch.stack([s['features'] for s in samples], dim=0)  # [B, N, C]
    
    captions = [s['caption'] for s in samples]
    valid_lengths = [s['valid_length'] for s in samples]

    # Build conversation
    conversations = create_conversation(
        questions="Translate sign language signal to Chinese.",
        answers=['']
    )

    # Insert wave tokens
    conversations = insert_wave_tokens(
        conversations,
        wave_token_lens=valid_lengths,
        wave_patch_token=model.model.config.default_wave_patch_token,
        wave_start_token=model.model.config.default_wave_start_token,
        wave_end_token=model.model.config.default_wave_end_token
    )

    # Prepare input and labels according to conv_type
    if conv_type == 'role':
        processed = prepare_conversation_data(conversations, tokenizer)
    elif conv_type == 'simple':
        processed = prepare_simple_data(conversations, tokenizer)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")

    input_ids = processed['input_ids'].to(wave_embeds.device)
    attention_mask = processed['attention_mask'].to(wave_embeds.device)
    labels = processed['labels'].to(wave_embeds.device)

    return input_ids, attention_mask, wave_embeds, labels

def generate_outputs(model, tokenizer, input_ids, attention_mask, wave_embeds):
    model.eval()
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.model.generate(
                input_ids=input_ids.cuda(),
                input_wave_embeds=wave_embeds.to(torch.bfloat16).cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=True,
                max_new_tokens=128,
                num_beams=5,
                top_k=50,
                top_p=0.95,
            )
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [text.strip() for text in generated_text]


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load configuration and initialize model
    cfg = load_yaml(args.config)
    
    # Initialize data module
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = 1
    data = instantiate_from_config(data_cfg)
    data.setup('fit')
    dataset = data.train_dataset
    
    # Initialize model
    with torch.autocast('cuda', dtype=torch.bfloat16):  
        model_cfg = cfg.model_cfg
        model_cfg.params.cfg.training.batch_size = 1
        model = instantiate_from_config(model_cfg)
    
    # Load checkpoint using load_state_dict_from_zero_checkpoint
    model = load_state_dict_from_zero_checkpoint(model, args.checkpoint)
    model.eval()

    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized in the model.")

    # Get conv_type from model config, default to 'role'
    conv_type = model_cfg.params.cfg.conv_type
    model_type = model_cfg.params.cfg.model_type
    
    # Select random samples
    total_samples = len(dataset)
    selected_indices = random.sample(range(total_samples), min(args.num_samples, total_samples))
    
    print("\nRunning inference on randomly selected samples...")
    print("-" * 80)
    
    with torch.no_grad():
        for idx in selected_indices:
            sample = dataset[idx]
            input_ids, attention_mask, wave_embeds, labels = prepare_batch([sample], model, tokenizer, conv_type=conv_type)
            outputs = generate_outputs(model, tokenizer, input_ids, attention_mask, wave_embeds)
            print("\nSample", idx)
            print(colored("Generated Caption:", "green"), colored(outputs[0], "green"))
            print(colored("Ground Truth:", "yellow"), colored(sample['caption'], "yellow"))
            print("-" * 80)

if __name__ == '__main__':
    main()