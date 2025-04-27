import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from typing import Dict, Any, List, Optional, Union
from src.model.llm.model_factory import ModelFactory
from src.model.llm.conversation import default_conversation
from src.utils.tools import instantiate_from_config
from termcolor import colored
from src.model.text_processor import preprocess_multimodal_wave, preprocess, format_conversation
IGNORE_INDEX = -100


class WaveLLM(pl.LightningModule):
    """PyTorch Lightning module for PEFT fine-tuning with LoRA"""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        
        # Create model using model factory
        self.model = ModelFactory.create_model(self.cfg.model_type, self.cfg.model)
        
        # Create tokenizer using model factory
        self.tokenizer = ModelFactory.create_tokenizer(self.cfg.model)
        
        # Initialize model tokens with tokenizer
        self.model.initialize_tokenizer_wave_backbone_config(
            tokenizer=self.tokenizer,
            device=self.device,
            fix_llm=self.cfg.fix_llm
        )
        
        # Initialize LoRA if enabled
        if self.cfg.use_peft:
            # First fix all parameters
            self.model.requires_grad_(False)

            # First get all linear layer names excluding mm_projection_layers
            target_modules = self._find_all_linear_names(
                exclude_keywords=['mm_projection_layers'])

            lora_config = LoraConfig(
                r=self.cfg.peft_config.r,
                lora_alpha=self.cfg.peft_config.lora_alpha,
                target_modules=target_modules,  # Use the filtered target modules
                lora_dropout=self.cfg.peft_config.lora_dropout,
                bias=self.cfg.peft_config.bias,
                task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, lora_config)

            # Enable training for mm_projection_layers
            self.model.get_model().mm_projection_layers.requires_grad_(True)

        # from IPython import embed; embed()
        # # Iterate over all parameters in the model and print their names and requires_grad status
        # for name, param in self.model.get_model().named_parameters():
        #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        
        # Print trainable parameters
        self._print_trainable_parameters()
    
    def _find_all_linear_names(self, exclude_keywords=None):
        """Find all linear layer names in the model"""
        if exclude_keywords is None:
            exclude_keywords = []
            
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if any(keyword in name for keyword in exclude_keywords):
                continue
            if isinstance(module, cls):
                lora_module_names.add(name)

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
    
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters in the model"""
        trainable_params = 0
        all_param = 0
        for name, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        trainable_params_str = colored(f"trainable params: {trainable_params}", 'green')
        all_params_str = colored(f"all params: {all_param}", 'yellow')
        trainable_percent_str = colored(f"trainable%: {100 * trainable_params / all_param:.2f}", 'blue')
        print(f"{trainable_params_str} || {all_params_str} || {trainable_percent_str}")

    def forward(self, batch):
        # Get wave embeddings
        wave_embeds = batch['features'].to(torch.bfloat16)  # [B, T, C]

        # # debug
        # wave_embeds = torch.zeros_like(wave_embeds)
        
        # Construct wave tokens
        B, T = wave_embeds.shape[:2]

        # Format conversations
        conversations = [format_conversation(
            question="Translate this millimeter wave signal to text.",
            answer=batch['caption'][i]
        ) for i in range(B)]

        # Get wave patch token from model config
        conversations = preprocess_multimodal_wave(
            conversations,
            wave_token_len=T,
            default_wave_patch_token=self.model.config.default_wave_patch_token,
            default_wave_start_token=self.model.config.default_wave_start_token,
            default_wave_end_token=self.model.config.default_wave_end_token
        )
        
        # Process text
        processed = preprocess(conversations, self.tokenizer)

        # Forward pass
        outputs = self.model(
            input_wave_embeds=wave_embeds,
            input_ids=processed['input_ids'].to(wave_embeds.device),
            attention_mask=processed['input_ids'].ne(self.tokenizer.pad_token_id).to(wave_embeds.device),
            labels=processed['labels'].to(wave_embeds.device)
        )

        # Decode labels and logits to text
        labels = processed['labels']; ignore_index = processed['labels'] == IGNORE_INDEX
        labels[ignore_index] = self.tokenizer.pad_token_id
        labels_decoded = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        logits = torch.argmax(outputs['logits'], dim=-1)
        logits[ignore_index] = self.tokenizer.pad_token_id
        logits_decoded = self.tokenizer.batch_decode(logits, skip_special_tokens=True)

        # # Print decoded texts with different colors
        # for label_text, logits_text in zip(labels_decoded, logits_decoded):
        #     print("-" * 80)
        #     print(colored("Label Text:", "yellow"), colored(label_text, "yellow"))
        #     print(colored("Logits Text:", "green"), colored(logits_text, "green"))

        return {
            'inputs_embeds': wave_embeds, 
            'attention_mask': processed['input_ids'].ne(self.tokenizer.pad_token_id),
            'loss': outputs['loss']
        }
    
    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens=128, num_beams=5):
        # Prepare model input
        model_input = {
            'input_wave_embeds': pre_compute_item['inputs_embeds'],
            'attention_mask': pre_compute_item['attention_mask']
        }
        
        # Generate text
        out = self.model.generate(
            model_input,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams
        )
        return out

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=False, 
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs['loss']
        self.log('valid/loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.training.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
