import os
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from termcolor import colored
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import sys
sys.path.append('.')

from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from src.model.llm.text_processor import preprocess_multimodal_wave, preprocess, format_conversation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--save_dir', type=str, default='attention_vis', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    return parser.parse_args()

def prepare_single_sample(sample, model, tokenizer):
    wave_embeds = sample['features'].unsqueeze(0)
    conversation = format_conversation(
        question="Translate this millimeter wave signal to text.",
        answer=sample['caption']
    )
    
    processed_conv = preprocess_multimodal_wave(
        [conversation],
        wave_token_len=wave_embeds.shape[1],
        default_wave_patch_token=model.config.default_wave_patch_token,
        default_wave_start_token=model.config.default_wave_start_token,
        default_wave_end_token=model.config.default_wave_end_token
    )
    
    processed = preprocess(processed_conv, tokenizer)
    input_ids = processed['input_ids'].to(wave_embeds.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    labels = processed['labels'].to(wave_embeds.device)
    
    # Print sequence information
    print("\nInput Sequence Analysis:")
    print("-" * 50)
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # for i, (token, id) in enumerate(zip(decoded_tokens, input_ids[0].cpu().numpy())):
    #     print(f"Position {i}: Token = {token}, ID = {id}")
    
    return input_ids, attention_mask, wave_embeds, decoded_tokens, labels

def get_attention_weights(model, input_ids, attention_mask, wave_embeds, labels):
    model.train()  # Use training mode
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            input_wave_embeds=wave_embeds.to(torch.bfloat16).cuda(),
            labels=labels.cuda(),
            output_attentions=True,
            return_dict=True
        )
        
        # Get the attention weights from the last layer
        last_layer_attention = outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]
        
        # Focus on the label part (i.e., the tokens we want to predict)
        label_positions = (labels != -100).squeeze()
        label_attention = last_layer_attention[0, :, label_positions, :]  # [num_heads, num_labels, seq_len]
        
        # Average over all attention heads
        avg_label_attention = label_attention.mean(dim=0)  # [num_labels, seq_len]
        
        return avg_label_attention.float().cpu().numpy()

def visualize_token_contributions(attention_weights, tokens, save_path, title, wave_start_idx=None, wave_end_idx=None):
    """
    Visualize the contribution of each input token to the prediction
    attention_weights: shape [num_labels, seq_len]
    """
    plt.figure(figsize=(20, 6))
    
    # Calculate average attention weight for each token across all prediction positions
    token_importance = attention_weights.mean(axis=0)  # [seq_len]
    
    # Create bar chart without text labels
    x = np.arange(len(tokens))
    bars = plt.bar(x, token_importance, width=1.0)
    
    # Set colors for different regions
    for i, bar in enumerate(bars):
        if wave_start_idx <= i <= wave_end_idx:
            bar.set_color('#FF6B6B')  # Red for wave region
        elif i < wave_start_idx:
            bar.set_color('#4ECDC4')  # Cyan for prompt region
        else:
            bar.set_color('#45B7D1')  # Blue for label region
    
    # Remove x-axis labels and ticks
    plt.xticks([])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', label='Prompt Region'),
        Patch(facecolor='#FF6B6B', label='Wave Region'),
        Patch(facecolor='#45B7D1', label='Label Region')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.ylabel('Average Attention Weight')
    plt.box(False)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_token_attention_flow(attention_weights, tokens, save_path, title, wave_start_idx=None, wave_end_idx=None):
    """Visualize attention flow between tokens"""
    plt.figure(figsize=(20, 3))
    
    # Calculate attention strength between token pairs
    attention_map = attention_weights[0]
    num_tokens = len(tokens)
    
    # Create table to display token relationships
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.set_axis_off()
    
    # Get maximum attention target for each token
    max_attention_targets = np.argmax(attention_map, axis=1)
    max_attention_values = np.max(attention_map, axis=1)
    
    # Create color mapping for different token types
    token_colors = []
    for i in range(num_tokens):
        if wave_start_idx <= i <= wave_end_idx:
            token_colors.append('#ffcccc')  # light red for wave
        elif i < wave_start_idx:
            token_colors.append('#cce5ff')  # light blue for input
        else:
            token_colors.append('#ccffcc')  # light green for output
    
    # Simplify table data to show positions and attention values
    table_data = [[f"Pos {i}\n↓\nPos {max_attention_targets[i]}\n({max_attention_values[i]:.3f})" for i in range(num_tokens)]]
    
    # Create and format table
    table = ax.table(cellText=table_data,
                    cellColours=[token_colors],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Adjust cell size and font
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for cell in table._cells:
        table._cells[cell].set_height(0.3)
    
    # Set title
    plt.title(title, pad=20)
    
    # Save visualization
    plt.savefig(save_path.replace('.png', '_flow.png'), dpi=300, bbox_inches='tight')
    plt.close()

def find_wave_token_indices(input_ids, wave_start_token, wave_end_token):
    """Find the indices of wave start and end tokens in the input sequence"""
    input_ids = input_ids.cpu().numpy()[0]  # Convert to numpy array and take first batch
    wave_start_idx = np.where(input_ids == wave_start_token)[0][0]
    wave_end_idx = np.where(input_ids == wave_end_token)[0][0]
    return wave_start_idx, wave_end_idx

def analyze_wave_attention(attention_weights, tokens, wave_start_idx, wave_end_idx):
    """Analyze the attention distribution of wave tokens and other regions"""
    # Calculate average attention to different regions for each prediction position
    wave_region_attention = attention_weights[:, wave_start_idx:wave_end_idx].mean(axis=1)
    prompt_region_attention = attention_weights[:, :wave_start_idx].mean(axis=1)
    label_region_attention = attention_weights[:, wave_end_idx:].mean(axis=1)
    
    print("\nDetailed Attention Analysis:")
    print("-" * 50)
    
    # Print statistics for each region
    print("\n1. Average Attention to Each Region:")
    print(f"Attention to Prompt: {prompt_region_attention.mean():.4f}")
    print(f"Attention to Wave: {wave_region_attention.mean():.4f}")
    print(f"Attention to Previous Labels: {label_region_attention.mean():.4f}")
    
    # Print max attention values
    print("\n2. Maximum Attention Values:")
    print(f"Max attention to Wave region: {wave_region_attention.max():.4f}")
    print(f"Max attention to Prompt region: {prompt_region_attention.max():.4f}")
    print(f"Max attention to Label region: {label_region_attention.max():.4f}")

def visualize_prediction_attention(attention_data, tokens, save_path, title, wave_start_idx, wave_end_idx):
    """Visualize attention distribution for each prediction position"""
    attention_weights = attention_data['attention_weights']
    valid_positions = attention_data['valid_positions']
    
    # Divide tokens into three regions
    token_regions = {
        'prompt': slice(0, wave_start_idx),
        'wave': slice(wave_start_idx, wave_end_idx + 1),
        'label': slice(wave_end_idx + 1, None)
    }
    
    # Calculate average attention contribution for each region
    region_contributions = {}
    for region_name, region_slice in token_regions.items():
        region_attention = attention_weights[:, :, region_slice].mean(axis=2)
        region_contributions[region_name] = region_attention.mean()
    
    # Create bar chart to show each region's contribution
    plt.figure(figsize=(10, 6))
    regions = list(region_contributions.keys())
    contributions = list(region_contributions.values())
    
    bars = plt.bar(regions, contributions)
    for bar, contribution in zip(bars, contributions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{contribution:.3f}',
                ha='center', va='bottom')
    
    plt.title(f'Region Contributions to Next Token Prediction\nLoss: {attention_data["loss"]:.4f}')
    plt.ylabel('Average Attention Weight')
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load configuration and model
    cfg = load_yaml(args.config)
    model_cfg = cfg.model_cfg
    model_cfg.params.cfg.batch_size = 1
    model = instantiate_from_config(model_cfg)
    
    # Load checkpoint
    model = load_state_dict_from_zero_checkpoint(model, args.checkpoint)
    model.eval()
    model = model.bfloat16()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get tokenizer
    tokenizer = model.tokenizer
    
    # Load dataset
    data_cfg = cfg.data_cfg
    data_cfg.params.cfg.batch_size = 1
    data = instantiate_from_config(data_cfg)
    data.setup('fit')
    dataset = data.train_dataset
    
    # Randomly select samples
    total_samples = len(dataset)
    selected_indices = random.sample(range(total_samples), min(args.num_samples, total_samples))
    
    for idx in selected_indices:
        sample = dataset[idx]
        input_ids, attention_mask, wave_embeds, tokens, labels = prepare_single_sample(sample, model.model, tokenizer)
        
        # Get attention weights
        attention_weights = get_attention_weights(model.model, input_ids, attention_mask, wave_embeds, labels)
        
        # Find wave token positions using input_ids directly
        wave_start_idx, wave_end_idx = find_wave_token_indices(
            input_ids,
            model.model.config.wave_start_token,
            model.model.config.wave_end_token
        )
        
        # Generate two types of visualizations
        base_name = f'attention_sample_{idx}'
        
        # 1. Token contribution visualization
        save_path = os.path.join(args.save_dir, f'{base_name}_contribution.png')
        visualize_token_contributions(
            attention_weights, 
            tokens, 
            save_path, 
            f'Token Contribution Analysis - Sample {idx}',
            wave_start_idx,
            wave_end_idx
        )
        
        # Analyze the attention distribution of wave tokens
        analyze_wave_attention(attention_weights, tokens, wave_start_idx, wave_end_idx)
        
        print(f"\nVisualizations saved to:")
        print(f"1. {os.path.join(args.save_dir, f'{base_name}_contribution.png')}")
        print("-" * 80)

if __name__ == '__main__':
    main() 