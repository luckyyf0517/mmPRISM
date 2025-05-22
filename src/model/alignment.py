import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.model.encoder.pose_encoder import HandPoseEncoder
from termcolor import colored

class AlignmentTrainer(pl.LightningModule):
    """Trainer for aligning predicted pose embeddings with ground truth pose embeddings"""
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        
        # Initialize both encoders with same weights
        self.gt_encoder = HandPoseEncoder(hidden_dim=64, output_dim=768)  # output_dim matches wavellm hidden size
        self.pred_encoder = HandPoseEncoder(hidden_dim=64, output_dim=768, additional_transformer=True)
        
        # Load pre-trained weights
        state_dict = torch.load(cfg.encoder_weights)
        self.gt_encoder.load_state_dict(state_dict)
        self.pred_encoder.load_state_dict(state_dict, strict=False)
        
        # Freeze ground truth encoder
        for param in self.gt_encoder.parameters():
            param.requires_grad = False
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Print trainable parameters
        self._print_trainable_parameters()
        
    def _print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        trainable_params_str = colored(f"trainable params: {trainable_params}", 'green')
        all_params_str = colored(f"all params: {all_param}", 'yellow')
        trainable_percent_str = colored(f"trainable%: {100 * trainable_params / all_param:.2f}", 'blue')
        print(f"{trainable_params_str} || {all_params_str} || {trainable_percent_str}")

    def forward(self, batch):
        # Get ground truth and predicted poses
        gt_pose = batch['joints_gt'].to(torch.bfloat16)  # [B, T, 2, 24, 3]
        valid_mask = ~torch.any(torch.isnan(gt_pose), dim=-1)
        gt_pose[~valid_mask] = 0

        pred_pose = batch['joints'].to(torch.bfloat16)   # [B, T, 2, 24, 3]

        # Get embeddings from both encoders
        with torch.no_grad():
            gt_embeds = self.gt_encoder(gt_pose)    # [B, T, hidden_size]
        pred_embeds = self.pred_encoder(pred_pose)  # [B, T, hidden_size]
        
        # Calculate loss
        loss = self.criterion(pred_embeds, gt_embeds)
        
        return {
            'loss': loss,
            'gt_embeds': gt_embeds,
            'pred_embeds': pred_embeds
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
        return optimizer