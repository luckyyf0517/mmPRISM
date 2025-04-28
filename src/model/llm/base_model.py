import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from transformers import PreTrainedModel

class WaveBaseModel(PreTrainedModel):
    """Base class for wave-enabled language models"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # Initialize wave projection layer
        self.mm_projection_layers = nn.Linear(self.config.mm_input_dim, self.config.hidden_size)
        
    def process_wave_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        input_wave_embeds: torch.Tensor,
    ) -> torch.FloatTensor:
        """Process and merge wave features with text embeddings"""
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        wave_features = self.mm_projection_layers(input_wave_embeds)

        new_input_embeds = []
        cur_wave_idx = 0
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            cur_wave_features = wave_features[cur_wave_idx].to(device=cur_input_embeds.device)
            num_patches = cur_wave_features.shape[0]
            
            if (cur_input_ids == self.config.wave_start_token).sum() != (cur_input_ids == self.config.wave_end_token).sum():
                raise ValueError("The number of wave start tokens and wave end tokens should be the same.")
            wave_start_tokens = torch.where(cur_input_ids == self.config.wave_start_token)[0]
            assert len(wave_start_tokens) > 0, "No wave start token found."
            for wave_start_token_pos in wave_start_tokens:
                if cur_input_ids[wave_start_token_pos + num_patches + 1] != self.config.wave_end_token:
                    raise ValueError("The wave end token should follow the wave start token.")
                if orig_embeds_params is not None:
                    cur_new_input_embeds = torch.cat((
                        cur_input_embeds[:wave_start_token_pos].detach(),
                        cur_input_embeds[wave_start_token_pos:wave_start_token_pos+1],
                        cur_wave_features,
                        cur_input_embeds[wave_start_token_pos + num_patches + 1:wave_start_token_pos + num_patches + 2],
                        cur_input_embeds[wave_start_token_pos + num_patches + 2:].detach()
                    ), dim=0)
                else:
                    cur_new_input_embeds = torch.cat((
                        cur_input_embeds[:wave_start_token_pos+1],
                        cur_wave_features,
                        cur_input_embeds[wave_start_token_pos + num_patches + 1:]
                    ), dim=0)
                cur_wave_idx += 1
            new_input_embeds.append(cur_new_input_embeds)
            
        return torch.stack(new_input_embeds, dim=0)

    def initialize_wave_tokens(
        self,
        tokenizer,
        device: str,
        fix_llm: bool = True
    ) -> None:
        """Initialize wave-related tokens and embeddings"""
        config = self.config
        
        # Add wave start/end tokens
        default_wave_start_token = config.default_wave_start_token
        default_wave_end_token = config.default_wave_end_token
        self.config.default_wave_start_token = default_wave_start_token
        self.config.default_wave_end_token = default_wave_end_token

        num_new_tokens = tokenizer.add_tokens([default_wave_start_token, default_wave_end_token], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        self.config.wave_start_token = tokenizer.convert_tokens_to_ids([default_wave_start_token])[0]
        self.config.wave_end_token = tokenizer.convert_tokens_to_ids([default_wave_end_token])[0]

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            # Set embedding trainability
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            if fix_llm:
                self.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
                print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
            else:
                self.orig_embeds_params = None
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True
                print("Setting output embeddings and all input embeddings trainable.")

    def prepare_inputs_for_generation(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for text generation"""
        input_wave_embeds = input_data.get('input_wave_embeds')
        attention_mask = input_data.get('attention_mask')
        
        if input_wave_embeds is not None:
            inputs_embeds = self.mm_projection_layers(input_wave_embeds)
        else:
            inputs_embeds = input_data.get('inputs_embeds')
            
        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        model_inputs.update(kwargs)
        return model_inputs