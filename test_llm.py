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
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_batch(tokenizer, conv_type='role', questions='', device='cuda'):
    # Stack features and collect captions/valid_lengths
    # Build conversation
    conversations = create_conversation(
        questions=questions,
        answers=[''], 
        add_wave=False
    )

    # Prepare input and labels according to conv_type
    if conv_type == 'role':
        processed = prepare_conversation_data(conversations, tokenizer)
    elif conv_type == 'simple':
        processed = prepare_simple_data(conversations, tokenizer)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")

    input_ids = processed['input_ids'].to(device)
    attention_mask = processed['attention_mask'].to(device)
    labels = processed['labels'].to(device)
    
    return input_ids, attention_mask, labels

def generate_outputs(model, tokenizer, input_ids, attention_mask):
    model.eval()
    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model.model.generate(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                max_new_tokens=128,
                num_beams=5,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95
            )
    # outputs = outputs[:, len(input_ids[0]):]
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return [text.strip() for text in generated_text]

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load configuration and initialize model
    cfg = load_yaml(args.config)
    
    # Initialize model
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = 1
    model = instantiate_from_config(model_cfg)
    
    # Load checkpoint using load_state_dict_from_zero_checkpoint
    if args.checkpoint:
        model = load_state_dict_from_zero_checkpoint(model, args.checkpoint)
    model.eval()
    model = model.bfloat16()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else None
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not initialized in the model.")

    # Get conv_type from model config, default to 'role'
    conv_type = 'role'
    
    # Select random samples
    print("\nRunning inference on randomly selected samples...")
    print("-" * 80)
    
    with torch.no_grad():
        def conversation_test(question):
            input_ids, attention_mask, labels = prepare_batch(tokenizer, conv_type=conv_type, questions=question)
            outputs = generate_outputs(model, tokenizer, input_ids, attention_mask)
            print("Generated Caption:\n", colored(outputs[0], "green"))
            print("-" * 80)

        from IPython import embed; embed()
        conversation_test('你是谁')

if __name__ == '__main__':
    main()