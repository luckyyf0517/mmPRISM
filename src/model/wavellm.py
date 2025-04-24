import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from typing import Dict, Any, List, Optional, Union
from src.model.llm.model_factory import ModelFactory
from src.utils.tools import instantiate_from_config


class WaveLLM(pl.LightningModule):
    """PyTorch Lightning module for PEFT fine-tuning with LoRA"""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        
        # Initialize simulator
        self.simulator = instantiate_from_config(self.cfg.simulator)
        
        # Initialize backbone
        self.backbone = instantiate_from_config(self.cfg.backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
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
            # First get all linear layer names excluding mm_projection_layers
            target_modules = self._find_all_linear_names(
                exclude_keywords=['mm_projection_layers']
            )
            
            lora_config = LoraConfig(
                r=self.cfg.peft_config.r,
                lora_alpha=self.cfg.peft_config.lora_alpha,
                target_modules=target_modules,  # Use the filtered target modules
                lora_dropout=self.cfg.peft_config.lora_dropout,
                bias=self.cfg.peft_config.bias,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            
            # Fix LLM parameters if specified
            if self.cfg.fix_llm:
                # First fix all parameters
                self.model.requires_grad_(False)
                
                # Then selectively enable training for mm_projection_layers
                if hasattr(self.model, 'get_model'):
                    # Some models wrap the base model in a get_model() method
                    if hasattr(self.model.get_model(), 'mm_projection_layers'):
                        self.model.get_model().mm_projection_layers.requires_grad_(True)
                elif hasattr(self.model, 'mm_projection_layers'):
                    self.model.mm_projection_layers.requires_grad_(True)
        
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
                print(f"{name}: {param.numel()} parameters")
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def forward(self, batch):
        # Directly use pre-computed features from dataset
        wave_embeds = batch['features'].to(torch.bfloat16)  # [B, T, C]
        
        # Get wave patch token from model config
        wave_start_token = self.model.config.default_wave_start_token
        wave_end_token = self.model.config.default_wave_end_token
        wave_patch_token = self.model.config.default_wave_patch_token

        # Construct input sequence
        B, T = wave_embeds.shape[:2]
        input_text = [f"Translate millimeter wave signal to text: {wave_start_token}{wave_patch_token*T}{wave_end_token}"] * B
        
        # Tokenize input sequence
        model_inputs = self.tokenizer(
            input_text,
            padding="longest",
            truncation=True,
            max_length=self.cfg.model.model_max_length,
            return_tensors="pt",
        ).to(wave_embeds.device)
        
        # Prepare target tokens
        target_tokens = self.tokenizer(
            batch['caption'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.model.model_max_length,
        ).to(wave_embeds.device)
        
        labels = target_tokens['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        input_length = model_inputs['input_ids'].size(1)
        labels = target_tokens['input_ids']
        if labels.size(1) < input_length:
            padding = torch.full(
                (labels.size(0), input_length - labels.size(1)),
                self.tokenizer.pad_token_id,
                device=labels.device)
            labels = torch.cat([labels, padding], dim=1)
        else: 
            labels = labels[:, :input_length]
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = self.model(
            input_wave_embeds=wave_embeds,
            input_ids=model_inputs['input_ids'],
            attention_mask=model_inputs['attention_mask'],
            labels=labels.to(wave_embeds.device)
        )
        
        return {
            'inputs_embeds': wave_embeds,
            'attention_mask': model_inputs['attention_mask'],
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
