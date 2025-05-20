import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration
from typing import Dict, Any, Optional, Union, List, Tuple
from torch.nn import CrossEntropyLoss
from src.model.llm.base_model import WaveBaseModel
import warnings

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
)

class MT5ForConditionalGeneration(WaveBaseModel, MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def get_output_embeddings(self):
        # Return lm_head for output embeddings
        return getattr(self, "lm_head", None)

    def forward(
        self,
        wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # Process input embeddings
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            
        # If wave_embeds is provided, concatenate it directly to text embeddings
        if wave_embeds is not None:
            inputs_embeds = self.process_wave_features(inputs_embeds, wave_embeds)
            # Update attention mask
            if attention_mask is not None:
                # Create all-ones mask for pose features (full attention)
                wave_mask = torch.ones(
                    (wave_embeds.shape[0], wave_embeds.shape[1]), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, wave_mask], dim=1)
        
        results = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # If labels are provided, manually implement loss calculation with label smoothing
        if labels is not None:
            logits = results.logits
            loss_fct = torch.nn.CrossEntropyLoss(
                label_smoothing=0.2,
                ignore_index=-100
            )
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            results['loss'] = loss

        return results

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        wave_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ):
        """
        Simplified generation function: directly concatenate features for generation
        """
        # Ensure at least one input type is provided
        assert wave_embeds is not None or input_ids is not None, \
            "Either wave_embeds or input_ids must be provided for generation"
        
        # If input_ids is provided, convert to embeddings
        if input_ids is not None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
        else:
            # If no text input, use an empty sequence
            inputs_embeds = torch.zeros(
                (wave_embeds.shape[0], 1, self.config.hidden_size), 
                dtype=wave_embeds.dtype, 
                device=wave_embeds.device
            )
            # Empty sequence attention mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], 1),
                    dtype=torch.long,
                    device=inputs_embeds.device
                )
        
        # Concatenate pose features
        if wave_embeds is not None:
            inputs_embeds = self.process_wave_features(inputs_embeds, wave_embeds)
            # Update attention mask
            if attention_mask is not None:
                wave_mask = torch.ones(
                    (wave_embeds.shape[0], wave_embeds.shape[1]), 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, wave_mask], dim=1)
        
        # Call parent's generate method
        return super().generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **model_kwargs
        )
