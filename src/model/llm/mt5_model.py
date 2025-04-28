import torch
import torch.nn as nn
from transformers import MT5Model, MT5ForConditionalGeneration
from typing import Dict, Any, Optional, Union, List, Tuple
from torch.nn import CrossEntropyLoss
from src.model.llm.base_model import WaveBaseModel

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
)

class MT5ModelWrapper(WaveBaseModel, MT5Model):
    def __init__(self, config):
        super().__init__(config)
        self.post_init()

    def forward(
        self,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)

        if input_ids.shape[1] != 1 or self.training:
            inputs_embeds = self.process_wave_features(input_ids, inputs_embeds, input_wave_embeds)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

class MT5ForConditionalGeneration(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = MT5ModelWrapper(config)
        self.post_init()

    def get_model(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        model_inputs.update({
            "input_wave_embeds": kwargs.get("input_wave_embeds", None),
        })
        return model_inputs

    def forward(
        self,
        input_wave_embeds: torch.Tensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        if labels is not None:
            decoder_input_ids = self.model._shift_right(labels)
        
        outputs = self.model(
            input_ids=input_ids,
            input_wave_embeds=input_wave_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        sequence_output = outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(self, input_data: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Generate text"""
        model_inputs = self.model.prepare_inputs_for_generation(input_data, **kwargs)
        return super().generate(**model_inputs)

    def initialize_tokenizer_wave_backbone_config(self, tokenizer, device, fix_llm=True):
        self.model.initialize_wave_tokens(tokenizer, device, fix_llm)