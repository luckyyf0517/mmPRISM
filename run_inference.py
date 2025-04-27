import os
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
from src.model.text_processor import preprocess_multimodal_wave, preprocess, format_conversation

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

def build_prompt(sample, model):
    # Construct a conversation consistent with training
    question = "Translate this millimeter wave signal to text."
    conversation = format_conversation(question=question, answer='')
    return conversation

def prepare_batch(samples, model, tokenizer):
    # wave_embeds: [B, T, C]
    wave_embeds = torch.stack([s['features'] for s in samples], dim=0)
    conversations = [build_prompt(s, model) for s in samples]

    # Retrieve tokens
    wave_start_token = model.config.default_wave_start_token
    wave_end_token = model.config.default_wave_end_token
    wave_patch_token = model.config.default_wave_patch_token

    processed_conv = preprocess_multimodal_wave(
        conversations,
        wave_token_len=wave_embeds.shape[1],
        default_wave_patch_token=wave_patch_token,
        default_wave_start_token=wave_start_token,
        default_wave_end_token=wave_end_token
    )
    processed = preprocess(processed_conv, tokenizer)
    input_ids = processed['input_ids'].to(wave_embeds.device)
    labels = processed['labels'].to(wave_embeds.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(wave_embeds.device)
    return input_ids, attention_mask, wave_embeds, labels

def generate_outputs(model, tokenizer, input_ids, attention_mask, wave_embeds):
    model.eval()
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=input_ids.cuda(),
                input_wave_embeds=wave_embeds.to(torch.bfloat16).cuda(),
                attention_mask=attention_mask.cuda(),
                do_sample=True,
                temperature=1.0,
                top_k=50,
                num_beams=5,
                max_new_tokens=128,
                top_p=0.95
            )
    input_token_len = input_ids.shape[1]
    generated_text = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)
    return [text.strip() for text in generated_text]

# def generate_outputs_forward(
#     model,
#     tokenizer,
#     input_ids,
#     attention_mask,
#     wave_embeds,
#     labels=None,
# ):
#     """
#     Directly use forward to output predictions for all tokens (non-generate), used for debugging input and label alignment issues.
#     """
#     model.eval()
#     with torch.no_grad():
#         with torch.autocast('cuda', dtype=torch.bfloat16):
#             outputs = model(
#                 input_ids=input_ids.cuda(),
#                 attention_mask=attention_mask.cuda(),
#                 input_wave_embeds=wave_embeds.cuda(),
#                 return_dict=True,
#             )
#             logits = outputs['logits']  # [B, L, Vocab]
#             ignore_index = (labels == -100)
#             pred_token_ids = logits.argmax(dim=-1)  # [B, L]
#             pred_token_ids[ignore_index] = tokenizer.pad_token_id
#             generated_text = tokenizer.batch_decode(pred_token_ids, skip_special_tokens=True)
#             return [text.strip() for text in generated_text]

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
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = 1
    model = instantiate_from_config(model_cfg)
    
    # Load checkpoint using load_state_dict_from_zero_checkpoint
    model = load_state_dict_from_zero_checkpoint(model, args.checkpoint)
    model.eval()
    model = model.bfloat16()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized in the model.")
    
    # Select random samples
    total_samples = len(dataset)
    selected_indices = random.sample(range(total_samples), min(args.num_samples, total_samples))
    
    print("\nRunning inference on randomly selected samples...")
    print("-" * 80)
    
    with torch.no_grad():
        for idx in selected_indices:
            sample = dataset[idx]
            wave_embeds = sample['features'].unsqueeze(0).to(model.device)
            input_ids, attention_mask, wave_embeds, labels = prepare_batch([sample], model.model, tokenizer)

            outputs = generate_outputs(model.model, tokenizer, input_ids, attention_mask, wave_embeds)

            # outputs = generate_outputs_forward(model.model, tokenizer, input_ids, attention_mask, wave_embeds, labels=labels)
            
            print("\nSample", idx)
            print(colored("Generated Caption:", "green"), colored(outputs[0], "green"))
            print(colored("Ground Truth:", "yellow"), colored(sample['caption'], "yellow"))
            print("-" * 80)

if __name__ == '__main__':
    main()
