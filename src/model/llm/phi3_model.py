import torch
import torch.nn as nn
from transformers import Phi3ForCausalLM, AutoTokenizer
from typing import List, Optional, Union, Dict, Any

class Phi3ModelWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_path = config.model_path
        
        # Initialize model and tokenizer
        self.model = Phi3ForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=config.model_max_length
        )
        
        # Add special tokens to tokenizer
        special_tokens = {
            'additional_special_tokens': [
                config.wave_start_token,
                config.wave_end_token,
                config.wave_patch_token
            ]
        }
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize projection layer
        self.mm_input_dim = config.mm_input_dim
        self.mm_projection_layers = nn.Linear(self.mm_input_dim, self.model.config.hidden_size)
        
        # Special tokens config
        self.config.wave_start_token = self.tokenizer.convert_tokens_to_ids(config.wave_start_token)
        self.config.wave_end_token = self.tokenizer.convert_tokens_to_ids(config.wave_end_token)
        self.config.wave_patch_token = self.tokenizer.convert_tokens_to_ids(config.wave_patch_token)
        self.config.mm_use_wave_start_end = getattr(config, 'mm_use_wave_start_end', True)

    def forward(
        self,
        input_wave_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)

        # Use input BNC features directly
        wave_features = self.mm_projection_layers(input_wave_embeds).to(dtype=torch.bfloat16)

        new_input_embeds = []
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            # Find start token position
            wave_start_pos = torch.where(cur_input_ids == self.config.wave_start_token)[0][0]
            # Get wave features for current sample
            cur_wave_features = wave_features[0].to(device=cur_input_embeds.device)
            num_patches = cur_wave_features.shape[0]
            
            # Check if end token is in correct position
            if cur_input_ids[wave_start_pos + num_patches + 1] != self.config.wave_end_token:
                raise ValueError("The wave end token should follow the wave start token.")
            
            # Concatenate features
            cur_new_input_embeds = torch.cat((
                cur_input_embeds[:wave_start_pos+1],  # Keep start token
                cur_wave_features,                     # Insert wave features
                cur_input_embeds[wave_start_pos + num_patches + 1:]  # Remaining part from end token
            ), dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            
        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        return outputs

    def generate(
        self,
        input_wave_tokens: Optional[torch.LongTensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_beams: int = 4,
        max_new_tokens: int = 30,
        **kwargs
    ) -> torch.LongTensor:
        """Generate text based on input tokens and wave embeddings."""
        
        # Prepare inputs_embeds if we have wave embeddings
        if input_wave_embeds is not None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            wave_features = self.mm_projection_layers(input_wave_embeds).to(dtype=torch.bfloat16)

            new_input_embeds = []
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                wave_start_pos = torch.where(cur_input_ids == self.config.wave_start_token)[0][0]
                cur_wave_features = wave_features[0].to(device=cur_input_embeds.device)
                num_patches = cur_wave_features.shape[0]
                
                cur_new_input_embeds = torch.cat((
                    cur_input_embeds[:wave_start_pos+1],
                    cur_wave_features,
                    cur_input_embeds[wave_start_pos + num_patches + 1:]
                ), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            input_ids = None  # Set to None since we're using inputs_embeds
        
        # Merge generation parameters
        generation_kwargs = {
            'input_ids': input_ids,
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'do_sample': do_sample,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
        }
        # Update with additional parameters without overwriting existing ones
        generation_kwargs.update(kwargs)
        
        # Generate with model
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(**generation_kwargs)
        
        return outputs
