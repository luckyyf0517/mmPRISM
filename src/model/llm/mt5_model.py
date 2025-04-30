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
        if inputs_embeds is None and input_ids is not None: # will not happen in inference
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            # Insert wave features into input_embeds
            inputs_embeds = self.process_wave_features(input_ids, inputs_embeds, input_wave_embeds)
        
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

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_wave_embeds: Optional[torch.Tensor] = None,
        **model_kwargs
    ):
        assert input_wave_embeds is not None and input_ids is not None, \
            "input_wave_embeds and input_ids must be provided"
        inputs_embeds = self.encoder.embed_tokens(input_ids)
        inputs_embeds = self.process_wave_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_wave_embeds=input_wave_embeds
        )
        return super().generate(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            **model_kwargs
        )
