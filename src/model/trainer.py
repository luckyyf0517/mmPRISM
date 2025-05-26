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
from src.model.encoder.pose_encoder import HandPoseEncoder
from src.utils.tools import instantiate_from_config
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from termcolor import colored
import json
IGNORE_INDEX = -100


class WaveLLMTrainer(pl.LightningModule):
    """PyTorch Lightning module for PEFT fine-tuning with LoRA"""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = self.cfg.training.batch_size
        
        # Create model
        self.model = ModelFactory.create_model(self.cfg.model_type, self.cfg.model)
        self.model.train()
        
        # Create pose encoder
        self.modalities = cfg.get('modalities', {'use_pred_pose': True, 'use_raw_pose': False})
        if self.modalities.get('use_pred_pose', False):
            input_dim = 3 if not self.enable_flow else 6
            self.hand_pose_encoder = HandPoseEncoder(input_dim=input_dim, hidden_dim=64, output_dim=self.model.config.hidden_size) # output_dim = 768
        else: 
            raise ValueError("Pose modality is not enabled")

        # Create tokenizer
        self.tokenizer = ModelFactory.create_tokenizer(self.cfg.model)
        
        # Initialize model
        self.model.initialize_wave_modules(
            device=self.device,
            fix_llm=self.cfg.fix_llm
        )
        
        # If LoRA is enabled, initialize LoRA
        if self.cfg.use_peft:
            # First, freeze all parameters
            self.model.requires_grad_(False)

            # Get all linear layer names
            target_modules = self._find_all_linear_names()

            lora_config = LoraConfig(
                r=self.cfg.peft_config.r,
                lora_alpha=self.cfg.peft_config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.cfg.peft_config.lora_dropout,
                bias=self.cfg.peft_config.bias,
                task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, lora_config)

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
    
    def _get_wave_embeds(self, joints):
        """
        Process pose data and generate embeddings. Optionally, concatenate with feature embeddings.
        Args:
            joints: [B, T, 2, 24, 3] Pose data
        Returns:
            wave_embeds: [B, T, hidden_size] Pose feature embeddings or concatenated embeddings
        """
        if self.modalities.get('use_pred_pose', False):
            if joints is None:
                raise ValueError("Pose modality is enabled but no pose data is provided")
            # Directly use HandPoseEncoder to encode pose, output aligned features
            pose_embeds = self.hand_pose_encoder(joints.to(torch.bfloat16))
        else:
            raise ValueError("Neither pose nor feature modality is enabled")
        return pose_embeds
    
    def _prepare_batch(self, batch):
        joints = batch.get('joints', None)
        valid_mask = ~torch.any(torch.isnan(joints), dim=-1)
        joints[~valid_mask] = 0
        batch['joints'] = joints
        return batch

    def forward(self, batch):
        # Get pose embeddings
        batch = self._prepare_batch(batch)
        wave_embeds = self._get_wave_embeds(batch['joints'])
        
        # Prepare prompt texts
        prompts = [f"Translate hand sign language videos to Chinese:" for _ in range(len(batch['caption']))]
        
        # Tokenize prompt texts
        prompt_tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(wave_embeds.device)
        
        # Tokenize target texts
        target_tokens = self.tokenizer(
            batch['caption'],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(wave_embeds.device)
        
        # Create labels, ignoring padding tokens
        labels = target_tokens['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        # Forward pass
        outputs = self.model(
            wave_embeds=wave_embeds,
            input_ids=prompt_tokens['input_ids'],
            attention_mask=prompt_tokens['attention_mask'],
            labels=labels
        )

        # with torch.no_grad():
        #     labels = labels.clone()
        #     labels[labels == IGNORE_INDEX] = self.tokenizer.pad_token_id
        #     references = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        #     predictions = self.tokenizer.batch_decode(outputs['logits'].argmax(dim=-1), skip_special_tokens=True)
        #     for i in range(len(references)):
        #         print('label: ', references[i])
        #         print('pred: ', predictions[i])
        #         print('-' * 50)

        return {
            'wave_embeds': wave_embeds, 
            'attention_mask': prompt_tokens['attention_mask'], 
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
            weight_decay=self.cfg.training.weight_decay, 
            betas=(0.9, 0.98)
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

    def on_test_epoch_start(self):
        """Initialize result JSON file"""
        # Initialize result file
        save_dir = os.path.join(self.cfg.ckpt_path, "evaluation")
        os.makedirs(save_dir, exist_ok=True)
        self.output_file = os.path.join(save_dir, f"results_rank_{self.global_rank}.json")
        # Create an empty dictionary and write to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        
        print(f"Initializing result file: {self.output_file}")

    def test_step(self, batch, batch_idx):
        """Generate translations and directly update JSON file"""
        # Use inference function to get prediction results
        preds = self._generate_translation(batch)
        # Get reference texts
        refs = batch['caption']
        # Get sample IDs
        sample_ids = batch.get('id')
        # Consistency check
        assert len(preds) == len(refs) and len(preds) == len(sample_ids), \
            f"Length mismatch in batch {batch_idx}. Predictions: {len(preds)}, References: {len(refs)}, IDs: {len(sample_ids)}"
        # Read current results
        with open(self.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # Add new results
        for i in range(len(sample_ids)):
            results[sample_ids[i]] = {
                "reference": refs[i],
                "prediction": preds[i],
            }
            print('-' * 50)
            print(colored("Reference text:", "blue"), refs[i])
            print(colored("Generated text:", "green"), preds[i])
        # Write updated results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        # Return None to save memory
        return None

    def _generate_translation(self, batch):
        """General translation generation function, can be called by test_step or inference script"""
        # Get pose features
        batch = self._prepare_batch(batch)
        wave_embeds = self._get_wave_embeds(batch['joints'])
        
        # Prepare prompt texts
        prompts = [f"Translate hand sign language videos to Chinese:" for _ in range(wave_embeds.shape[0])]
        
        # Tokenize prompt texts
        prompt_tokens = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(wave_embeds.device)
        
        # Execute generation
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=prompt_tokens['input_ids'],
                attention_mask=prompt_tokens['attention_mask'],
                wave_embeds=wave_embeds,
                do_sample=False,
                max_new_tokens=128,
                num_beams=4,
            )
        
        # Decode and return all generated texts
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
