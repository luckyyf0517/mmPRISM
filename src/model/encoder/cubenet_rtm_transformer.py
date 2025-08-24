import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.encoder.cubenet_rtm import CSPEncoder3D
from src.utils.tools import get_obj_from_str


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences
    
    Adds learnable positional embeddings to frame features to encode temporal order.
    """
    def __init__(self, d_model, max_len=16):
        super().__init__()
        self.d_model = d_model
        
        # Create learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, d_model]
        
        Returns:
            torch.Tensor: Position-encoded tensor [B, T, d_model]
        """
        B, T, D = x.shape
        
        # Add positional encoding
        pos_emb = self.pos_embedding[:, :T, :]  # [1, T, d_model]
        x = x + pos_emb  # Broadcasting: [B, T, D] + [1, T, D]
        
        return x


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder for temporal feature aggregation
    
    Uses multi-head self-attention to model temporal dependencies between frames
    and aggregates information through a learnable CLS token.
    """
    def __init__(self, 
                 d_model=1024,
                 nhead=8, 
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 max_temporal_frames=16):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Positional encoding for temporal order
        self.pos_encoding = PositionalEncoding(d_model, max_temporal_frames + 1)  # +1 for CLS token
        
        # CLS token - learnable global aggregation token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Input projection (if needed)
        self.input_projection = nn.Linear(d_model, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize transformer weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, frame_features):
        """
        Args:
            frame_features (torch.Tensor): Frame features [B, T, d_model]
        
        Returns:
            torch.Tensor: Aggregated features from CLS token [B, d_model]
        """
        B, T, D = frame_features.shape
        
        # Project input features
        frame_features = self.input_projection(frame_features)  # [B, T, d_model]
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        
        # Concatenate CLS token with frame features
        # CLS token is placed at the beginning of the sequence
        sequence = torch.cat([cls_tokens, frame_features], dim=1)  # [B, T+1, d_model]
        
        # Add positional encoding
        sequence = self.pos_encoding(sequence)  # [B, T+1, d_model]
        
        # Apply transformer encoder
        encoded_sequence = self.transformer_encoder(sequence)  # [B, T+1, d_model]
        
        # Extract CLS token (first token) as the aggregated representation
        cls_output = encoded_sequence[:, 0, :]  # [B, d_model]
        
        # Apply output projection
        output_features = self.output_projection(cls_output)  # [B, d_model]
        
        return output_features


class CSPRTMTransformerEncoder3D(nn.Module):
    """CSP-RTM Encoder with Transformer-based temporal aggregation
    
    Architecture:
    1. Shared CSPEncoder3D extracts spatial features from each frame
    2. Transformer encoder with CLS token aggregates temporal information
    3. Output projection generates final feature representation
    
    Key advantages:
    - Self-attention captures long-range temporal dependencies
    - CLS token provides learnable global aggregation
    - Positional encoding preserves temporal order information
    - Scalable to different sequence lengths
    """
    def __init__(self,
                 # Spatial encoder parameters
                 in_channels=64,
                 base_channels=64,
                 stage_channels=[128, 256, 512, 1024],
                 stage_blocks=[2, 4, 4, 2],
                 spp_kernel_sizes=(3, 5, 7),
                 expansion=0.5,
                 channel_attention=True,
                 norm_layer='torch.nn.BatchNorm3d',
                 spatial_encoder_pretrained=None,
                 spatial_encoder_freeze=False,
                 # Temporal transformer parameters
                 temporal_frames=5,
                 transformer_layers=6,
                 transformer_heads=8,
                 transformer_dim_feedforward=2048,
                 transformer_dropout=0.1,
                 # Output parameters
                 output_dim=256,
                 **kwargs):
        super().__init__()
        
        self.temporal_frames = temporal_frames
        self.spatial_feature_dim = stage_channels[-1]  # Feature dimension from spatial encoder
        self.output_dim = output_dim
        
        # Shared spatial encoder for all frames
        self.spatial_encoder = CSPEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            spp_kernel_sizes=spp_kernel_sizes,
            expansion=expansion,
            channel_attention=channel_attention,
            norm_layer=norm_layer,
            use_csp=True,
            use_channel_attention=True,
            use_pafpn=True
        )
        
        # Load pretrained weights for spatial encoder if provided
        if spatial_encoder_pretrained is not None:
            self._load_spatial_encoder_pretrained(spatial_encoder_pretrained)
        
        # Freeze spatial encoder if requested
        if spatial_encoder_freeze:
            self._freeze_spatial_encoder()
        
        # Temporal transformer encoder
        self.temporal_transformer = TemporalTransformerEncoder(
            d_model=self.spatial_feature_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            max_temporal_frames=temporal_frames
        )
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.spatial_feature_dim, self.spatial_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(transformer_dropout),
            nn.Linear(self.spatial_feature_dim // 2, output_dim)
        )
        
        # Feature dimension for compatibility
        self.feature_dim = output_dim
        
        self._init_weights()
    
    def _load_spatial_encoder_pretrained(self, pretrained_path):
        """Load pretrained weights for the spatial encoder"""
        import os
        
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pretrained weights file not found: {pretrained_path}")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Filter state dict to only include spatial encoder parameters
            spatial_encoder_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith('spatial_encoder.'):
                    # Remove 'spatial_encoder.' prefix
                    new_key = key[len('spatial_encoder.'):]
                    spatial_encoder_state_dict[new_key] = value
                elif not any(prefix in key for prefix in ['temporal_transformer', 'output_projection']):
                    # If no prefix, assume it's directly from CSPEncoder3D
                    spatial_encoder_state_dict[key] = value
            
            # Load the filtered state dict
            missing_keys, unexpected_keys = self.spatial_encoder.load_state_dict(
                spatial_encoder_state_dict, strict=False
            )
            
            if missing_keys:
                print(f"Missing keys in spatial encoder: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in spatial encoder: {unexpected_keys}")
            
            print(f"Successfully loaded pretrained weights for spatial encoder from: {pretrained_path}")
            print(f"Loaded {len(spatial_encoder_state_dict)} parameters")
            
        except Exception as e:
            print(f"Error loading pretrained weights for spatial encoder: {e}")
            print("Continuing with random initialization...")
    
    def _freeze_spatial_encoder(self):
        """Freeze the spatial encoder parameters"""
        print("Freezing spatial encoder parameters...")
        
        frozen_params = 0
        for name, param in self.spatial_encoder.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        print(f"Frozen {frozen_params:,} parameters in spatial encoder")
        self.spatial_encoder.eval()
    
    def _unfreeze_spatial_encoder(self):
        """Unfreeze the spatial encoder parameters"""
        print("Unfreezing spatial encoder parameters...")
        
        unfrozen_params = 0
        for name, param in self.spatial_encoder.named_parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        print(f"Unfrozen {unfrozen_params:,} parameters in spatial encoder")
        self.spatial_encoder.train()
    
    def get_trainable_parameters(self):
        """Get information about trainable vs frozen parameters"""
        spatial_trainable = sum(p.numel() for p in self.spatial_encoder.parameters() if p.requires_grad)
        spatial_total = sum(p.numel() for p in self.spatial_encoder.parameters())
        
        transformer_trainable = sum(p.numel() for p in self.temporal_transformer.parameters() if p.requires_grad)
        transformer_total = sum(p.numel() for p in self.temporal_transformer.parameters())
        
        output_trainable = sum(p.numel() for p in self.output_projection.parameters() if p.requires_grad)
        output_total = sum(p.numel() for p in self.output_projection.parameters())
        
        total_trainable = spatial_trainable + transformer_trainable + output_trainable
        total_params = spatial_total + transformer_total + output_total
        
        return {
            'spatial_encoder': {
                'trainable': spatial_trainable,
                'total': spatial_total,
                'frozen': spatial_total - spatial_trainable
            },
            'temporal_transformer': {
                'trainable': transformer_trainable,
                'total': transformer_total,
                'frozen': transformer_total - transformer_trainable
            },
            'output_projection': {
                'trainable': output_trainable,
                'total': output_total,
                'frozen': output_total - output_trainable
            },
            'total': {
                'trainable': total_trainable,
                'total': total_params,
                'frozen': total_params - total_trainable
            }
        }
    
    def train(self, mode=True):
        """Override train method to handle frozen spatial encoder"""
        super().train(mode)
        
        # If spatial encoder is frozen, keep it in eval mode
        if not any(p.requires_grad for p in self.spatial_encoder.parameters()):
            self.spatial_encoder.eval()
        
        return self
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with Transformer-based temporal aggregation
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, 64, 32, 32, 32]
        
        Returns:
            global_feat (torch.Tensor): Global feature vector [B, output_dim]
        """
        B, T, C, H, W, D = x.shape
        
        # Extract spatial features from each frame using shared encoder
        frame_features = []
        for t in range(T):
            frame = x[:, t, :, :, :, :]  # [B, C, H, W, D]
            frame_feat = self.spatial_encoder(frame)  # [B, spatial_feature_dim]
            frame_features.append(frame_feat)
        
        # Stack frame features: [B, T, spatial_feature_dim]
        temporal_features = torch.stack(frame_features, dim=1)
        
        # Apply transformer-based temporal aggregation
        # The transformer uses CLS token to aggregate information from all frames
        aggregated_features = self.temporal_transformer(temporal_features)  # [B, spatial_feature_dim]
        
        # Final output projection
        global_feat = self.output_projection(aggregated_features)  # [B, output_dim]
        
        return global_feat 