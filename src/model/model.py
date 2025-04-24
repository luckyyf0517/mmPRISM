import torch
import torch.nn as nn
from torch import Tensor
import math
from transformers import AutoTokenizer, AutoConfig
import warnings
from pytorch_lightning import LightningModule
from src.utils.tools import instantiate_from_config
from src.model.llm.model_factory import ModelFactory
from timm.models.layers import trunc_normal_


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class mmWave2Text(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        
        # Initialize signal encoder
        self.signal_encoder = instantiate_from_config(cfg.signal_encoder)
        
        # Initialize weights
        self.apply(self._init_weights)

        # Prepare model configuration
        model_config = {
            'model_path': cfg.model_path,
            'mm_input_dim': cfg.signal_encoder.params.output_channels,
            'model_max_length': cfg.model_max_length
        }
        
        # Create model using model factory
        self.model = ModelFactory.create_model(cfg.model_type, model_config)
        self.tokenizer = self.model.get_tokenizer()
        
        # Initialize loss function
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing, ignore_index=-100)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, src_input, tgt_input):
        # Process input signal [B, N, C]
        B, N, C = src_input.shape
        inputs_embeds = self.signal_encoder(src_input) # [B, N, C]
        
        # Add prefix token
        prefix_token = self.tokenizer(
            [f"Translate millimeter wave signal to text: "] * B,
            padding="longest",
            truncation=True,
            max_length=self.cfg.model_max_length,
            return_tensors="pt",
        ).to(inputs_embeds.device)
        
        prefix_embeds = self.model.get_model().embed_tokens(prefix_token['input_ids'])
        
        # Prepare attention mask
        attention_mask = torch.cat([
            prefix_token['attention_mask'],
            torch.ones((B, N), device=inputs_embeds.device)
        ], dim=1)
        
        # Prepare target tokens
        tgt_input_tokenizer = self.tokenizer(
            tgt_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        
        labels = tgt_input_tokenizer['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Prepare model input
        model_input = {
            'input_wave_embeds': inputs_embeds,
            'inputs_embeds': prefix_embeds,
            'attention_mask': attention_mask,
            'labels': labels.to(inputs_embeds.device)
        }
        
        # Forward pass
        outputs = self.model.forward(model_input)
        
        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
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
        outputs = self(batch['signal'], batch['caption'])
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=False, 
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['signal'], batch['caption'])
        loss = outputs['loss']
        self.log('valid/loss', loss, on_step=False, on_epoch=True, 
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }
