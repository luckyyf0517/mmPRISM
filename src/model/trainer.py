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
from src.model.encoder.pose_encoder import HandPoseEncoder
from src.utils.tools import instantiate_from_config
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
        
        self.modalities = cfg.modalities
    
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

    def _get_wave_embeds(self, features, joints):
        """Process and merge enabled modalities
        Args:
            features: [B, T, feature_dim] or None
            joints: [B, T, 2, 24, 3] or None
        Returns:
            wave_embeds: [B, T, total_dim]
        """
        embeds_list = []
        
        # Process features if enabled
        if self.modalities['use_features']:
            if features is None:
                raise ValueError("Features modality is enabled but no features provided")
            feature_embeds = features.to(torch.bfloat16)  # [B, T, 512]
            embeds_list.append(feature_embeds)
        
        # Process pose if enabled
        if self.modalities['use_pred_pose']:
            if joints is None:
                raise ValueError("Pose modality is enabled but no joints provided")
            pose_embeds = self.hand_pose_encoder(joints.to(torch.bfloat16))  # [B, T, 768]
            embeds_list.append(pose_embeds)
        
        # Merge all enabled modalities
        if len(embeds_list) == 0:
            raise ValueError("No modalities enabled")
        elif len(embeds_list) == 1:
            wave_embeds = embeds_list[0]
        else:
            wave_embeds = torch.cat(embeds_list, dim=-1)  # [B, T, total_dim]
        
        return wave_embeds

    def forward(self, batch):
        # Get wave embeddings based on enabled modalities
        features = batch.get('features', None)
        joints = batch.get('joints', None)
        wave_embeds = self._get_wave_embeds(features, joints)
        
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

    def on_test_epoch_start(self):
        """Initialize JSON file for results"""
        # Initialize the results file
        save_dir = os.path.join(self.cfg.ckpt_path, "evaluation")
        os.makedirs(save_dir, exist_ok=True)
        self.output_file = os.path.join(save_dir, f"results_rank_{self.global_rank}.json")
        # Create an empty dictionary and write it to file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
        
        print(f"Initialized results file: {self.output_file}")

    def test_step(self, batch, batch_idx):
        """Generate translations and directly update JSON file"""
        # Use the general inference function to get prediction results
        preds = self._generate_translation(batch)
        
        # Get the reference text
        refs = batch['caption']
        
        # Get sample ID
        sample_ids = batch.get('id')
        
        # Consistency check
        assert len(preds) == len(refs) and len(preds) == len(sample_ids), \
            f"Length mismatch in batch {batch_idx}. Predictions: {len(preds)}, References: {len(refs)}, IDs: {len(sample_ids)}"
        
        # Read current results
        with open(self.output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Add new result
        for i in range(len(sample_ids)):
            results[sample_ids[i]] = {
                "prediction": preds[i],
                "reference": refs[i]
            }

            print('-' * 100)
            print(colored("Ground truth:", "blue"), refs[i])
            print(colored("Generated text:", "green"), preds[i])
        
        # Write updated results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Return nothing to save memory
        return None

    def _generate_translation(self, batch):
        """General translation generation function, can be called by test_step or inference script"""
        # Get wave features
        wave_embeds = self._get_wave_embeds(
            batch.get('features', None), 
            batch.get('joints', None)
        )
        
        # Prepare input data
        input_ids, attention_mask = self._prepare_inference_inputs(
            wave_embeds, 
            batch['valid_length']
        )
        
        # Execute generation
        return self._inference_generate(input_ids, attention_mask, wave_embeds)

    def _prepare_inference_inputs(self, wave_embeds, valid_lengths):
        """Prepare input data required for inference"""
        # Construct conversation
        conversations = create_conversation(
            questions="Translate sign language signal to Chinese.",
            answers=[''] * self.batch_size  # No answer provided during inference
        )
        
        # Insert wave tokens
        conversations = insert_wave_tokens(
            conversations,
            wave_token_lens=valid_lengths,
            wave_patch_token=self.model.config.default_wave_patch_token,
            wave_start_token=self.model.config.default_wave_start_token,
            wave_end_token=self.model.config.default_wave_end_token
        )
        
        # Process data based on conversation type
        if self.conv_type == 'role':
            processed = prepare_conversation_data(conversations, self.tokenizer)
        elif self.conv_type == 'simple':
            processed = prepare_simple_data(conversations, self.tokenizer)
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")
        
        # Move data to the correct device
        input_ids = processed['input_ids'].to(wave_embeds.device)
        attention_mask = processed['attention_mask'].to(wave_embeds.device)
        
        return input_ids, attention_mask

    def _inference_generate(self, input_ids, attention_mask, wave_embeds):
        """Execute model generation"""
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                input_wave_embeds=wave_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=128,
                num_beams=5,
                top_k=50,
                top_p=0.95,
            )
        
        # Decode and return ALL generated texts (don't just take index 0)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _save_predictions(self, preds, refs, save_dir="./outputs"):
        """Save prediction and reference results to a file"""
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Save predictions and references
        pred_path = os.path.join(save_dir, "test_predictions.txt")
        ref_path = os.path.join(save_dir, "test_references.txt")
        
        with open(pred_path, "w", encoding="utf-8") as f_pred, \
             open(ref_path, "w", encoding="utf-8") as f_ref:
            for pred, ref in zip(preds, refs):
                f_pred.write(pred + "\n")
                f_ref.write(ref + "\n")
        
        print(f"Predictions and references saved to {pred_path} and {ref_path}")

