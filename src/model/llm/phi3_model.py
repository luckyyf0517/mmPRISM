import torch
import torch.nn as nn
from transformers import Phi3Model, Phi3ForCausalLM
from typing import List, Optional, Union, Dict, Any, Tuple
from torch.nn import CrossEntropyLoss
from src.model.llm.base_model import WaveBaseModel

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

class Phi3ForCausalLM(WaveBaseModel, Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def get_output_embeddings(self):
        # Return lm_head for output embeddings
        return getattr(self, "lm_head", None)

    def forward(
        self,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # Prepare input embeddings
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
        # Insert wave features into input_embeds
        if input_wave_embeds is not None:
            inputs_embeds = self.process_wave_features(input_ids, inputs_embeds, input_wave_embeds)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=0,
        **kwargs,
    ):
        # For Phi3, which is a causal LLM, we need to retain input_ids for generation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs
        )
        
        # Add input_wave_embeds to model_inputs
        model_inputs["input_wave_embeds"] = kwargs.get("input_wave_embeds", None)
        return model_inputs