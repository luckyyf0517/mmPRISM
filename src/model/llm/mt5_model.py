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
        input_wave_embeds: torch.Tensor = None,
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
        # Prepare input embeddings
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)

            if input_wave_embeds is not None:   
                # Insert wave features into input_embeds
                inputs_embeds = self.process_wave_features(input_ids, inputs_embeds, input_wave_embeds)
            else: 
                warnings.warn("input_wave_embeds is not provided, using input_ids to generate wave features")
        
        return super().forward(
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is None:  # First call
            # Process encoder inputs
            if inputs_embeds is None and input_ids is not None:
                encoder_inputs_embeds = self.encoder.embed_tokens(input_ids)
            # Process wave embeddings
            if kwargs["input_wave_embeds"] is not None: 
                encoder_inputs_embeds = self.process_wave_features(
                    input_ids=input_ids,
                    inputs_embeds=encoder_inputs_embeds,
                    input_wave_embeds=kwargs["input_wave_embeds"]
                )
            # Run encoder
            encoder_outputs = self.encoder(
                inputs_embeds=encoder_inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            kwargs["encoder_outputs"] = encoder_outputs
        
        # Call the parent class's prepare_inputs_for_generation
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids, 
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs
        )
        return model_inputs