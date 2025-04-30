import sys
sys.path.append('.')

import os
import torch
import torch.nn as nn
from pprint import pprint
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from src.model.llm.model_factory import ModelFactory
from src.model.llm.text_processor import *
from src.utils.tools import instantiate_from_config
from termcolor import colored
IGNORE_INDEX = -100


class WaveLLMTrainer(pl.LightningModule):
    """PyTorch Lightning module for PEFT fine-tuning with LoRA"""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        
        # Create model using model factory
        self.model = ModelFactory.create_model(self.cfg.model_type, self.cfg.model)
        self.hand_pose_encoder = HandPoseEncoder()
        
        # Create tokenizer using model factory
        self.tokenizer = ModelFactory.create_tokenizer(self.cfg.model)
        self.conv_type = self.cfg.conv_type # 'role' or 'simple'
        
        # Initialize model tokens with tokenizer
        self.model.initialize_wave_tokens(
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
        trainable_params_str = colored(f"trainable params: {trainable_params}", 'green')
        all_params_str = colored(f"all params: {all_param}", 'yellow')
        trainable_percent_str = colored(f"trainable%: {100 * trainable_params / all_param:.2f}", 'blue')
        print(f"{trainable_params_str} || {all_params_str} || {trainable_percent_str}")

    def forward(self, batch):
        # Get wave embeddings
        # wave_embeds = batch['features'].to(torch.bfloat16)  # [B, T, C]

        poses = batch['joints'].to(torch.bfloat16)  # [B, T, 2, 24, 3]
        wave_embeds = self.hand_pose_encoder(poses)  # [B, T, C]
        
        # Format conversations
        conversations = create_conversation(
            questions="Translate sign language signal to Chinese.",
            answers=batch['caption']
        )

        # Get wave patch token from model config
        conversations = insert_wave_tokens(
            conversations,
            wave_token_lens=batch['valid_length'],
            wave_patch_token=self.model.config.default_wave_patch_token,
            wave_start_token=self.model.config.default_wave_start_token,
            wave_end_token=self.model.config.default_wave_end_token
        )

        # Prepare input and labels
        if self.conv_type == 'role':
            # input: [<prompt>, <question>, <wave>, <answer>]
            # labels: [<masked_prompt>, <masked_wave>, <masked_question>, <answer>]
            processed = prepare_conversation_data(conversations, self.tokenizer)
        elif self.conv_type == 'simple':
            # input: [<question>, <wave>]
            # labels: [<answer>]
            processed = prepare_simple_data(conversations, self.tokenizer)

        # Forward pass
        outputs = self.model(
            input_wave_embeds=wave_embeds,
            input_ids=processed['input_ids'].to(wave_embeds.device),
            attention_mask=processed['attention_mask'].to(wave_embeds.device),
            labels=processed['labels'].to(wave_embeds.device)
        )

        # from IPython import embed; embed()
        # pprint(self.tokenizer.batch_decode(processed['labels'], skip_special_tokens=True))
        # pprint(self.tokenizer.batch_decode(outputs['logits'].argmax(dim=-1), skip_special_tokens=True))

        return {
            'inputs_embeds': wave_embeds, 
            'attention_mask': processed['attention_mask'], 
            'loss': outputs['loss']
        }
    
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
            weight_decay=self.cfg.training.weight_decay
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


from src.model.stgcn_layers import Graph, get_stgcn_chain
class HandPoseEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Initialize graphs for body and hands
        self.modes = ['body', 'left', 'right']
        self.graph = {}
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        
        # Projection layer
        self.proj_linear = nn.Linear(3, hidden_dim)
        
        # Create graph and GCN for body and hands
        for mode in self.modes:
            if mode == 'body':
                self.graph[mode] = Graph(layout='body', strategy='distance', max_hop=1)
            else:
                self.graph[mode] = Graph(layout='hand', strategy='distance', max_hop=1)
            A = torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False)
            
            # Create spatial and temporal GCN modules
            spatial_kernel_size = A.size(0)
            self.gcn_modules[mode], final_dim = get_stgcn_chain(
                hidden_dim, 
                'spatial', 
                (1, spatial_kernel_size), 
                A.clone(), 
                True
            )
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                final_dim,
                'temporal',
                (5, spatial_kernel_size),
                A.clone(),
                True
            )

    def forward(self, x):
        """
        Input: x [B, N, 2, 24, 3] - batch, frames, hands(left/right), joints, coords
        Output: [B, N, C] - C is the final feature dimension
        """
        features = []
        
        # Reshape input data format
        x = {
            'body': torch.cat([x[:, :, 0, :3], x[:, :, 1, :3]], dim=2),  # Concatenate the first 3 points of both hands
            'left': x[:, :, 0, 3:],  # All points of the left hand
            'right': x[:, :, 1, 3:]  # All points of the right hand
        }
        
        # Process body features first
        body_data = x['body']  # [B, N, 6, 3]
        body_proj = self.proj_linear(body_data)
        body_proj = body_proj.permute(0, 3, 1, 2)  # [B, C, N, 6]
        body_feat = self.gcn_modules['body'](body_proj)
        body_feat = self.fusion_gcn_modules['body'](body_feat)
        # Add body features to output
        pool_body_feat = body_feat.mean(-1).transpose(1, 2)  # [B, N, C]
        features.append(pool_body_feat)
        # Process left and right hands
        for mode in ['left', 'right']:
            # Get data for one hand [B, N, 24, 3]
            hand_data = x[mode]
            # Project to hidden dim [B, N, 24, hidden_dim]
            proj_feat = self.proj_linear(hand_data)
            proj_feat = proj_feat.permute(0, 3, 1, 2)
            # Forward pass through spatial GCN
            spatial_feat = self.gcn_modules[mode](proj_feat)
            # Add body reference features
            if mode == 'left':
                ref_feat = body_feat[..., [2]]
            else:
                ref_feat = body_feat[..., [5]]
            spatial_feat = spatial_feat + ref_feat.detach()
            # Forward pass through temporal GCN
            temporal_feat = self.fusion_gcn_modules[mode](spatial_feat)
            # Average pooling over node dimension [B, C, N]
            pool_feat = temporal_feat.mean(-1)
            # Rearrange dimensions to [B, N, C]
            pool_feat = pool_feat.transpose(1, 2)
            features.append(pool_feat)
        
        # Merge features from both hands
        output = torch.cat(features, dim=-1)  # [B, N, C*2]
        return output
