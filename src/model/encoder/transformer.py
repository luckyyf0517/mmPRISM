import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.encoder.cubenet import CubeNet
from src.utils.tools import get_obj_from_str


class RoPEPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for improved temporal modeling
    
    RoPE provides better relative position awareness and generalizes better
    to sequences longer than those seen during training.
    """
    def __init__(self, d_model, max_len=64, theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create frequency tensor for rotary embeddings
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for computed rotary embeddings
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _compute_rope_cache(self, seq_len, device, dtype):
        """Compute and cache rotary embeddings"""
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        
        # Create position indices
        position = torch.arange(seq_len, device=device, dtype=dtype)
        
        # Compute frequencies
        freqs = torch.outer(position, self.inv_freq.to(dtype))
        
        # Create cos and sin embeddings
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Cache the results
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_seq_len = seq_len
        
        return cos, sin
    
    def apply_rotary_emb(self, x, cos, sin):
        """Apply rotary embeddings to input tensor"""
        # Split the last dimension into pairs
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model//2]
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Rotary transformation
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
    
    def forward(self, q, k, v):
        """
        Apply RoPE to query and key tensors
        
        Args:
            q, k, v: Query, key, value tensors [B, seq_len, d_model]
        
        Returns:
            Rotary-encoded q, k, v tensors
        """
        seq_len = q.size(1)
        cos, sin = self._compute_rope_cache(seq_len, q.device, q.dtype)
        
        # Apply RoPE only to query and key
        q_rope = self.apply_rotary_emb(q, cos, sin)
        k_rope = self.apply_rotary_emb(k, cos, sin)
        
        return q_rope, k_rope, v


class TemporalConvBlock(nn.Module):
    """1D Temporal convolution block for local temporal feature extraction
    
    Captures short-term temporal dependencies before global attention.
    """
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        
        self.conv1d = nn.Conv1d(
            d_model, d_model, 
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
            groups=d_model // 4  # Depthwise-separable convolution
        )
        
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        residual = x
        
        # Apply temporal convolution
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)  # [B, T, d_model]
        
        # Residual connection with normalization
        x = self.norm1(x + residual)
        
        return x


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feedforward network with gating mechanism
    
    SwiGLU shows better performance than traditional ReLU-based FFN.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        # SwiGLU components
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_model, d_ff, bias=False)  # Value projection  
        self.w3 = nn.Linear(d_ff, d_model, bias=False)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()  # Swish activation
        
    def forward(self, x):
        """
        SwiGLU: x -> SiLU(xW1) ⊙ (xW2) -> W3
        """
        gate = self.silu(self.w1(x))
        value = self.w2(x)
        hidden = gate * value  # Element-wise gating
        output = self.w3(self.dropout(hidden))
        return output


class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with RoPE, temporal conv, and SwiGLU"""
    
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, temporal_kernel=3):
        super().__init__()
        
        self.d_model = d_model
        # Ensure nhead divides d_model evenly
        self.nhead = self._get_valid_num_heads(d_model, nhead)
        
        # RoPE positional encoding
        self.rope = RoPEPositionalEncoding(d_model // nhead)
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Temporal convolution for local dependencies
        self.temporal_conv = TemporalConvBlock(d_model, temporal_kernel, dropout=dropout)
        
        # SwiGLU feedforward network
        self.ffn = SwiGLUFeedForward(d_model, d_ff, dropout)
        
        # Normalization layers (Pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Gating mechanism for feature fusion
        self.gate_conv = nn.Parameter(torch.ones(1))
        self.gate_attn = nn.Parameter(torch.ones(1))
        
        self.dropout = nn.Dropout(dropout)
        
    def _get_valid_num_heads(self, d_model, desired_heads):
        """Get a valid number of heads that divides d_model evenly"""
        # Find the largest divisor of d_model that is <= desired_heads
        for heads in range(desired_heads, 0, -1):
            if d_model % heads == 0:
                return heads
        return 1  # Fallback to 1 head
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, d_model]
            mask: Optional attention mask
        """
        B, T, D = x.shape
        
        # 1. Temporal convolution branch (local dependencies)
        conv_out = self.temporal_conv(self.norm1(x))
        
        # 2. Self-attention branch (global dependencies)
        attn_input = self.norm2(x)
        
        # Project to Q, K, V
        q = self.q_proj(attn_input).view(B, T, self.nhead, D // self.nhead)
        k = self.k_proj(attn_input).view(B, T, self.nhead, D // self.nhead)
        v = self.v_proj(attn_input).view(B, T, self.nhead, D // self.nhead)
        
        # Apply RoPE to each head separately
        q_rope_list = []
        k_rope_list = []
        v_list = []
        
        for h in range(self.nhead):
            q_h = q[:, :, h, :]  # [B, T, head_dim]
            k_h = k[:, :, h, :]
            v_h = v[:, :, h, :]
            
            q_rope_h, k_rope_h, v_h = self.rope(q_h, k_h, v_h)
            q_rope_list.append(q_rope_h)
            k_rope_list.append(k_rope_h)
            v_list.append(v_h)
        
        # Recombine heads
        q_rope = torch.stack(q_rope_list, dim=2)  # [B, T, nhead, head_dim]
        k_rope = torch.stack(k_rope_list, dim=2)
        v = torch.stack(v_list, dim=2)
        
        # Reshape for attention computation
        q_rope = q_rope.transpose(1, 2).contiguous().view(B * self.nhead, T, D // self.nhead)
        k_rope = k_rope.transpose(1, 2).contiguous().view(B * self.nhead, T, D // self.nhead)
        v = v.transpose(1, 2).contiguous().view(B * self.nhead, T, D // self.nhead)
        
        # Scaled dot-product attention
        scale = (D // self.nhead) ** -0.5
        attn_weights = torch.bmm(q_rope, k_rope.transpose(1, 2)) * scale
        
        if mask is not None:
            attn_weights.masked_fill_(mask.unsqueeze(0).expand(B * self.nhead, -1, -1), float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_out = torch.bmm(attn_weights, v)
        attn_out = attn_out.view(B, self.nhead, T, D // self.nhead)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)
        
        # 3. Gated fusion of conv and attention features
        fused_features = self.gate_conv * conv_out + self.gate_attn * attn_out
        x = x + self.dropout(fused_features)
        
        # 4. SwiGLU feedforward
        x = x + self.dropout(self.ffn(self.norm3(x)))
        
        return x


class AdvancedTemporalTransformerEncoder(nn.Module):
    """Advanced Transformer encoder with state-of-the-art temporal modeling
    
    Key improvements:
    - RoPE (Rotary Position Embedding) for better position awareness
    - Temporal convolutions for local dependencies  
    - SwiGLU feedforward networks
    - Gated feature fusion
    - Cross-attention between spatial and temporal features
    - Enhanced CLS token aggregation
    """
    def __init__(self, 
                 d_model=1024,
                 nhead=8, 
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 max_temporal_frames=16,
                 temporal_kernel_size=3,
                 use_cross_attention=True):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_cross_attention = use_cross_attention
        
        # Enhanced CLS token with learnable temperature
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.cls_temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Input projection with residual connection
        self.input_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                d_ff=dim_feedforward,
                dropout=dropout,
                temporal_kernel=temporal_kernel_size
            ) for _ in range(num_layers)
        ])
        
        # Cross-attention for spatial-temporal feature fusion
        if use_cross_attention:
            # Ensure nhead divides d_model evenly
            cross_attn_heads = self._get_valid_num_heads(d_model, nhead)
            self.spatial_temporal_cross_attn = nn.MultiheadAttention(
                d_model, cross_attn_heads, dropout=dropout, batch_first=True
            )
            self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Enhanced output projection with multiple aggregation strategies
        self.output_strategies = nn.ModuleDict({
            'cls': nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            ),
            'mean_pool': nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model)
            ),
                         'attention_pool': nn.MultiheadAttention(
                 d_model, num_heads=1, dropout=dropout, batch_first=True
             )
        })
        
        # Learnable weights for aggregation fusion
        self.aggregation_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Initialize learnable query for attention pooling
        self.learnable_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self._init_weights()
    
    def _get_valid_num_heads(self, d_model, desired_heads):
        """Get a valid number of heads that divides d_model evenly"""
        # Find the largest divisor of d_model that is <= desired_heads
        for heads in range(desired_heads, 0, -1):
            if d_model % heads == 0:
                return heads
        return 1  # Fallback to 1 head
    
    def _init_weights(self):
        """Enhanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    if hasattr(m, 'out_features') and m.out_features == self.d_model:
                        # Output projections use smaller initialization
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize CLS token with smaller variance
        nn.init.trunc_normal_(self.cls_token, std=0.01)
        
        # Initialize learnable query for attention pooling
        nn.init.trunc_normal_(self.learnable_query, std=0.01)
    
    def forward(self, frame_features):
        """
        Args:
            frame_features (torch.Tensor): Frame features [B, T, d_model]
        
        Returns:
            torch.Tensor: Enhanced aggregated features [B, d_model]
        """
        B, T, D = frame_features.shape
        
        # Enhanced input projection with residual
        projected_features = self.input_projection(frame_features)
        frame_features = frame_features + projected_features
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        
        # Concatenate CLS token with frame features
        sequence = torch.cat([cls_tokens, frame_features], dim=1)  # [B, T+1, d_model]
        
        # Apply enhanced transformer layers
        for layer in self.transformer_layers:
            sequence = layer(sequence)
        
        # Extract CLS token and frame features
        cls_output = sequence[:, 0, :]  # [B, d_model]
        temporal_features = sequence[:, 1:, :]  # [B, T, d_model]
        
        # Cross-attention for spatial-temporal fusion (optional)
        if self.use_cross_attention:
            # Use CLS token as query, temporal features as key/value
            cls_query = cls_output.unsqueeze(1)  # [B, 1, d_model]
            cross_attn_out, _ = self.spatial_temporal_cross_attn(
                cls_query, temporal_features, temporal_features
            )
            cls_output = self.cross_attn_norm(cls_output + cross_attn_out.squeeze(1))
        
        # Multiple aggregation strategies
        # 1. CLS token strategy
        cls_feat = self.output_strategies['cls'](cls_output)
        
        # 2. Mean pooling strategy
        mean_feat = temporal_features.mean(dim=1)
        mean_feat = self.output_strategies['mean_pool'](mean_feat)
        
        # 3. Attention pooling strategy
        query = self.learnable_query.expand(B, -1, -1)
        attn_feat, _ = self.output_strategies['attention_pool'](
            query, temporal_features, temporal_features
        )
        attn_feat = attn_feat.squeeze(1)
        
        # Weighted fusion of aggregation strategies
        weights = F.softmax(self.aggregation_weights, dim=0)
        final_features = (weights[0] * cls_feat + 
                         weights[1] * mean_feat + 
                         weights[2] * attn_feat)
        
        return final_features


class CubeNetTransformer(nn.Module):
    """CubeNet Encoder with Advanced Transformer-based temporal aggregation
    
    Architecture enhancements:
    1. Shared CubeNet extracts spatial features from each frame
    2. Advanced Transformer encoder with RoPE, temporal convolutions, SwiGLU
    3. Cross-attention for spatial-temporal feature fusion
    4. Multiple aggregation strategies with learnable fusion
    5. Enhanced output projection with gating mechanisms
    
    Key advantages:
    - RoPE provides better relative position modeling
    - Temporal convolutions capture local dependencies
    - SwiGLU improves feature transformation quality
    - Cross-attention enables richer spatial-temporal interactions
    - Multiple aggregation strategies improve robustness
    """
    def __init__(self,
                 # Spatial encoder parameters
                 in_channels=64,
                 base_channels=64,
                 stage_channels=[128, 256, 512, 1024],
                 stage_blocks=[2, 4, 4, 2],
                 use_attention=True,
                 use_pafpn=True,
                 use_se_attention=True,
                 spatial_encoder_pretrained=None,
                 spatial_encoder_freeze=False,
                 # Enhanced temporal transformer parameters
                 temporal_frames=5,
                 transformer_layers=8,  # Increased layers for better modeling
                 transformer_heads=12,  # More heads for richer attention
                 transformer_dim_feedforward=3072,  # Larger FFN
                 transformer_dropout=0.1,
                 temporal_kernel_size=3,
                 use_cross_attention=True,
                 # Output parameters
                 output_dim=256,
                 **kwargs):
        super().__init__()
        
        self.temporal_frames = temporal_frames
        self.spatial_feature_dim = stage_channels[-1]
        self.output_dim = output_dim
        
        # Shared spatial encoder for all frames
        self.spatial_encoder = CubeNet(
            in_channels=in_channels,
            base_channels=base_channels,
            stage_channels=stage_channels,
            stage_blocks=stage_blocks,
            use_attention=use_attention,
            use_pafpn=use_pafpn,
            use_se_attention=use_se_attention
        )
        
        # Load pretrained weights for spatial encoder if provided
        if spatial_encoder_pretrained is not None:
            self._load_spatial_encoder_pretrained(spatial_encoder_pretrained)
        
        # Freeze spatial encoder if requested
        if spatial_encoder_freeze:
            self._freeze_spatial_encoder()
        
        # Advanced temporal transformer encoder
        self.temporal_transformer = AdvancedTemporalTransformerEncoder(
            d_model=self.spatial_feature_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            max_temporal_frames=temporal_frames,
            temporal_kernel_size=temporal_kernel_size,
            use_cross_attention=use_cross_attention
        )
        
        # Enhanced final output projection with gating
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.spatial_feature_dim),
            nn.Linear(self.spatial_feature_dim, self.spatial_feature_dim),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(self.spatial_feature_dim, self.spatial_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(transformer_dropout * 0.5),
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
                    # If no prefix, assume it's directly from CubeNet
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


# Alias for backward compatibility
CSPRTMTransformerEncoder3D = CubeNetTransformer 