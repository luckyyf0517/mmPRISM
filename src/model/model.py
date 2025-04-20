import torch
import torch.nn as nn
from torch import Tensor
import math
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import warnings
from pytorch_lightning import LightningModule

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
        
        # Signal processing layers
        self.signal_proj = nn.Linear(cfg.doppler_size, cfg.hidden_dim)
        
        # Temporal processing layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([cfg.hidden_dim, cfg.max_length]),
            nn.GELU(),
            nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([cfg.hidden_dim, cfg.max_length]),
            nn.GELU()
        )
        
        # Feature projection
        self.feature_proj = nn.Linear(cfg.hidden_dim, 768)
        
        # Initialize MT5 model and tokenizer
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(cfg.mt5_path)
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(
            cfg.mt5_path,
            legacy=False,
            model_max_length=512  # Set a reasonable maximum length
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, src_input, tgt_input):
        # Process input signal [B, T, N]
        B, T, N = src_input.shape
        self.batch_size = B
        
        # Project signal to hidden dimension
        signal = self.signal_proj(src_input)  # [B, T, hidden_dim]

        # Process temporal dimension
        signal = signal.permute(0, 2, 1)  # [B, hidden_dim, T]
        signal = self.temporal_conv(signal)  # [B, hidden_dim, T]
        signal = signal.permute(0, 2, 1)  # [B, T, hidden_dim]
        
        # Project each timestep to MT5 dimension
        inputs_embeds = self.feature_proj(signal)  # [B, T, 768]
        
        # Add prefix token with explicit max_length
        prefix_token = self.mt5_tokenizer(
            [f"Translate millimeter wave signal to text: "] * B,
            padding="longest",
            truncation=True,
            max_length=512,  # Explicitly specify maximum length
            return_tensors="pt",
        ).to(inputs_embeds.device)
        
        prefix_embeds = self.mt5_model.encoder.embed_tokens(prefix_token['input_ids'])
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)  # Directly concatenate all timesteps
        
        # Prepare attention mask - Modified to include all timesteps
        attention_mask = torch.cat([
            prefix_token['attention_mask'],
            torch.ones((B, T), device=inputs_embeds.device)  # Add mask for each timestep
        ], dim=1)
        
        # Prepare target tokens with explicit max_length
        tgt_input_tokenizer = self.mt5_tokenizer(
            tgt_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,  # Explicitly specify maximum length for target text
        )
        
        labels = tgt_input_tokenizer['input_ids']
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100
        
        # Forward through MT5
        out = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(inputs_embeds.device),
            return_dict=True
        )
        
        # Compute loss
        label = labels.reshape(-1)
        out_logits = out['logits']
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        loss_fct = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing, ignore_index=-100)
        loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))
        
        return {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'loss': loss
        }
        
    @torch.no_grad()
    def generate(self, pre_compute_item, max_new_tokens=128, num_beams=5):
        inputs_embeds = pre_compute_item['inputs_embeds']
        attention_mask = pre_compute_item['attention_mask']
        
        out = self.mt5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams)
        return out

    def training_step(self, batch, batch_idx):
        outputs = self(batch['signal'], batch['caption'])
        loss = outputs['loss']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['signal'], batch['caption'])
        loss = outputs['loss']
        self.log('valid/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.max_epochs,
            eta_min=0
        )
        return [optimizer], [scheduler]
