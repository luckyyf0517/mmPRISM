import os
import json
import warnings
from typing import Tuple, Union, Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import numpy as np
import pytorch_lightning as pl
import einops
from einops.layers.torch import Rearrange
from termcolor import colored
from easydict import EasyDict as edict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from timm.models.vision_transformer import _create_vision_transformer

from src.model.clip_loss import create_loss

# Suppress specific warning from transformers
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")


class TextEncoder(nn.Module):
    """
    Text encoder module that uses a pre-trained language model.
    
    Args:
        model_name: Name of the pre-trained model to use
        text_pooling: Method to pool text embeddings ('mean', 'pooler', or 'max')
        unfreeze_last_layer_num: Number of last layers to unfreeze for fine-tuning
    """
    def __init__(self, model_name: str, text_pooling: str = 'pooler', unfreeze_last_layer_num: int = 0, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.text_pooling = text_pooling
        self.unfreeze_last_layer_num = unfreeze_last_layer_num
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.text_encoder.eval()
        
        # Configure parameter freezing
        self._configure_parameter_freezing()
        
    def _configure_parameter_freezing(self):
        """Configure which parameters are frozen and which are trainable."""
        for name, param in self.text_encoder.named_parameters():
            num_layers = len(self.text_encoder.encoder.layer)
            unfreeze_param = False
            
            # Unfreeze specified number of last layers
            for i in range(self.unfreeze_last_layer_num): 
                if f'layer.{num_layers - i}' in name: 
                    unfreeze_param = True
                if 'pooler' in name: 
                    unfreeze_param = True
                    
            param.requires_grad = unfreeze_param
        
    def encode(self, text: Union[str, List[str]], device: str = 'cuda') -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            text: Input text or list of texts
            device: Device to place tensors on
            
        Returns:
            Text embeddings
        """
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        
        if self.text_pooling == 'mean': 
            out = outputs.last_hidden_state.mean(dim=1)
        elif self.text_pooling == 'pooler': 
            out = outputs.pooler_output
        elif self.text_pooling == 'max': 
            out = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown text pooling method: {self.text_pooling}")
            
        return out


class ImageEncoder(nn.Module):
    """
    Image encoder module that uses a pre-trained vision model.
    
    Args:
        model_name: Name of the pre-trained model to use
        embed_dim: Dimension of the output embedding
        image_resolution: Resolution of input images
        pretrained: Whether to use pre-trained weights
    """
    def __init__(
        self,
        model_name: str,
        embed_dim: int,
        image_resolution: int, 
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        # Override image resolution with fixed values
        image_resolution = [384, 64]
        self.model_name = model_name
        
        # Initialize appropriate vision model
        if 'vit' in model_name: 
            self.visual = _create_vision_transformer(
                model_name, 
                pretrained=pretrained, 
                in_chans=1, 
                img_size=image_resolution, 
                num_classes=embed_dim, 
                **kwargs
            )
        elif 'resnet' in model_name: 
            self.visual = torchvision.models.resnet18(pretrained=pretrained)
            # Modify for single channel input
            self.visual.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Modify output dimension
            self.visual.fc = nn.Linear(self.visual.fc.in_features, embed_dim)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the image encoder.
        
        Args:
            x: Input image tensor
            
        Returns:
            Image embeddings
        """
        return self.visual(x)
    

class CLIPProjection(nn.Module):
    """
    Projection module for CLIP embeddings.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to a new dimension.
        
        Args:
            x: Input embeddings
            
        Returns:
            Projected embeddings
        """
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class CLIP(pl.LightningModule):
    """
    CLIP (Contrastive Language-Image Pre-training) model.
    
    This implementation uses a transformer for time series encoding rather than text encoding.
    
    Args:
        encoder_cfg: Configuration for the image encoder
        text_cfg: Configuration for the text encoder
        context_length: Maximum context length for the transformer
        transformer_width: Width of the transformer
        transformer_layers: Number of transformer layers
        transformer_heads: Number of attention heads
        temperature: Temperature parameter for the contrastive loss
        use_siglip: Whether to use SigLIP loss instead of standard CLIP loss
        learning_rate: Learning rate for optimization
        max_epochs: Maximum number of training epochs
    """
    def __init__(
        self,
        encoder_cfg: dict,
        text_cfg: dict,
        temperature: float,
        use_siglip: bool = False,
        learning_rate: float = 1.0e-04,
        max_epochs: int = 50
    ):
        super().__init__()
        # Save configuration
        self.save_hyperparameters()
        
        # Initialize encoders
        self.image_encoder = ImageEncoder(**encoder_cfg)
        self.text_encoder = TextEncoder(**text_cfg)
        
        # Initialize projection layers
        self.text_projection = CLIPProjection(text_cfg.embed_dim, encoder_cfg.embed_dim)
        
        # Configure loss
        self.use_siglip = use_siglip
        if use_siglip: 
            self.logit_scale = nn.Parameter(torch.tensor(1.0))
            self.logit_bias = nn.Parameter(torch.tensor(-10.0))
        else:
            self.logit_scale = 1 / temperature
        
        # Initialize loss function
        loss_args = edict({
            'gather_with_grad': True,
            'rank': int(os.environ.get('RANK', 0)),
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'horovod': False, 
            'local_loss': False,
            'siglip': use_siglip,
            'loss_dist_impl': 'bidir',
        })
        self.loss_fn = create_loss(loss_args)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Initialize parameters and optimizers
        self.initialize_parameters()
        self.configure_optimizers()
        
    def initialize_parameters(self):
        """Initialize model parameters."""
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection.projection.weight, std=0.01)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into embeddings.
        
        Args:
            image: Input image tensor
            
        Returns:
            Image embeddings
        """
        return self.image_encoder(image)

    def encode_text(self, text: Union[str, List[str]], device: str = 'cuda') -> torch.Tensor:
        """
        Encode text into embeddings.
        
        Args:
            text: Input text or list of texts
            device: Device to place tensors on
            
        Returns:
            Text embeddings
        """
        x = self.text_encoder.encode(text, device=device)
        x = self.text_projection(x)
        return x

    def forward(self, image: torch.Tensor, text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the CLIP model.
        
        Args:
            image: Input image tensor
            text: Input text or list of texts
            
        Returns:
            Tuple of normalized image and text features
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, device=image.device)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True) 
        return image_features, text_features
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute the CLIP loss.
        
        Args:
            batch: Batch containing image and text features
            
        Returns:
            Dictionary with loss values
        """
        image_features = batch['image_features']
        text_features = batch['text_features']
        
        if not self.use_siglip: 
            loss_clip = self.loss_fn(
                image_features, 
                text_features, 
                logit_scale=self.logit_scale)
        else: 
            loss_clip = self.loss_fn(
                image_features, 
                text_features, 
                logit_scale=self.logit_scale, 
                logit_bias=self.logit_bias)
            
        return {'loss_clip': loss_clip}
    
    def shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, phase: str = 'train') -> Dict[str, torch.Tensor]:
        """
        Shared step for training, validation, and testing.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            phase: Current phase ('train', 'valid', or 'test')
            
        Returns:
            Processed batch with features
        """
        self.batch_size = batch['signal'].size(0)
        batch['signal'] = batch['signal'].float().contiguous()  # [b, c, h, w]
        
        signal = batch['signal']
        text = batch['caption']
        
        image_features, text_features = self.forward(signal, text)
        
        batch['image_features'] = image_features.contiguous()
        batch['text_features'] = text_features.contiguous()
        
        return batch
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        batch = self.shared_step(batch, batch_idx, phase='train')
        loss_dict = self.compute_loss(batch)
        self.log_loss(loss_dict, phase='train')
        return loss_dict['loss_clip']
        
    def on_validation_epoch_start(self):
        """Called when validation epoch starts."""
        self.on_test_epoch_start()
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        """
        Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            dataloader_idx: DataLoader index
        """
        batch = self.shared_step(batch, batch_idx, phase='valid')
        loss_dict = self.compute_loss(batch)
        self.log_loss(loss_dict, phase='valid')
        
    def on_validation_epoch_end(self):
        """Called when validation epoch ends."""
        self.on_test_epoch_end()
        
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            List of optimizers and schedulers
        """
        lr = self.learning_rate
        
        # Create optimizer with different learning rates for different components
        opt = torch.optim.AdamW([
            {'params': self.text_encoder.parameters(), 'lr': lr / 2},
            {'params': self.text_projection.parameters(), 'lr': lr},
            {'params': self.image_encoder.parameters(), 'lr': lr},
        ], betas=(0.5, 0.9), weight_decay=0.01)
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=self.max_epochs, 
            eta_min=0
        )
        
        return [opt], [scheduler]
    
    def log_loss(self, loss_dict: Dict[str, torch.Tensor], phase: str = 'train'):
        """
        Log loss values.
        
        Args:
            loss_dict: Dictionary of loss values
            phase: Current phase ('train', 'valid', or 'test')
        """
        for k, v in loss_dict.items(): 
            self.log(
                f"{phase}/{k}", 
                v, 
                on_step=self.training, 
                on_epoch=not self.training, 
                logger=True, 
                batch_size=self.batch_size, 
                rank_zero_only=True, 
                sync_dist=False, 
                add_dataloader_idx=False
            )
        del loss_dict
