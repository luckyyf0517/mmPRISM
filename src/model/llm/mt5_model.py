import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer, AutoConfig
from typing import Dict, Any, Optional, Union, List


class MT5ModelWrapper(nn.Module):
    """Wrapper class for MT5 model, implementing the MultiModalLanguageModel interface"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_path = config.model_path
        
        # Initialize model and tokenizer
        self.model = MT5ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_path,
            model_max_length=config.model_max_length
        )
        
        # Initialize wave-related components
        self.mm_input_dim = config.mm_input_dim
        self.mm_projection_layers = nn.Linear(self.mm_input_dim, self.model.config.d_model)
        
        # Special tokens config
        self.wave_start_token = config.wave_start_token
        self.wave_end_token = config.wave_end_token
        self.wave_patch_token = config.wave_patch_token
        self.mm_use_wave_start_end = getattr(config, 'mm_use_wave_start_end', True)

    def forward(
        self,
        input_wave_tokens: Optional[torch.LongTensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.shared(input_ids)

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if input_ids.shape[1] != 1 or self.training:
            if input_wave_tokens is not None or input_wave_embeds is not None:
                bs = input_ids.shape[0]
                wave_features = self.mm_projection_layers(input_wave_embeds)

                new_input_embeds = []
                cur_wave_idx = 0
                for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                    cur_wave_features = wave_features[cur_wave_idx].to(device=cur_input_embeds.device)
                    num_patches = cur_wave_features.shape[0]

                    if self.mm_use_wave_start_end:
                        if (cur_input_ids == self.wave_start_token).sum() != (cur_input_ids == self.wave_end_token).sum():
                            raise ValueError("The number of wave start tokens and wave end tokens should be the same.")
                        wave_start_tokens = torch.where(cur_input_ids == self.wave_start_token)[0]
                        for wave_start_token_pos in wave_start_tokens:
                            if cur_input_ids[wave_start_token_pos + num_patches + 1] != self.wave_end_token:
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
                    else:
                        if (cur_input_ids == self.wave_patch_token).sum() != num_patches:
                            raise ValueError("The number of wave patch tokens should be the same as the number of wave patches.")
                        masked_indices = torch.where(cur_input_ids == self.wave_patch_token)[0]
                        mask_index_start = masked_indices[0]
                        if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                            raise ValueError("The wave patch tokens should be consecutive.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_wave_features,
                                cur_input_embeds[mask_index_start+num_patches:].detach()
                            ), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:mask_index_start],
                                cur_wave_features,
                                cur_input_embeds[mask_index_start+num_patches:]
                            ), dim=0)
                        new_input_embeds.append(cur_new_input_embeds)
                        cur_wave_idx += 1
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
            else:
                raise ValueError("Either input_wave_tokens or input_wave_embeds should be provided.")

        # For MT5, we need decoder input ids
        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
        else:
            batch_size = inputs_embeds.shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=inputs_embeds.device
            )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        return outputs
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Generate text"""
        # Extract required parameters from input data
        input_wave_embeds = input_data.get('input_wave_embeds')
        attention_mask = input_data.get('attention_mask')
        
        # Process multimodal input
        if input_wave_embeds is not None:
            # Project multimodal features directly
            inputs_embeds = self.mm_projection_layers(input_wave_embeds)
        else:
            inputs_embeds = input_data.get('inputs_embeds')
        
        # Call model generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
    
    def get_tokenizer(self):
        """Get tokenizer"""
        return self.tokenizer
    
    def get_model(self):
        """Get underlying model"""
        return self.model