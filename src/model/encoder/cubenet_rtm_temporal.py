import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder.cubenet_rtm import CSPEncoder3D
from src.utils.tools import get_obj_from_str


class DopplerVelocityAccumulator(nn.Module):
    """Doppler-specific velocity accumulator that models velocity-to-spatial information flow
    
    Key innovation: Explicitly models how Doppler (velocity) information from earlier frames
    accumulates and influences spatial feature extraction in later frames, mimicking the
    physical principle that velocity changes affect spatial positioning over time.
    """
    def __init__(self, feature_dim, num_velocity_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_velocity_heads = num_velocity_heads
        self.head_dim = feature_dim // num_velocity_heads
        
        # Doppler velocity extraction - separates velocity components from spatial features
        self.velocity_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2),  # Compress to velocity-specific features
            nn.Tanh()  # Velocity can be positive or negative
        )
        
        # Spatial context extractor - maintains spatial information
        self.spatial_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # Multi-head Doppler attention - models how different velocity components
        # influence different spatial regions
        self.doppler_attention = nn.MultiheadAttention(
            embed_dim=feature_dim // 2,
            num_heads=num_velocity_heads,
            batch_first=True
        )
        
        # Velocity-to-spatial transformation - key innovation
        # Models how accumulated velocity affects spatial feature distribution
        self.velocity_to_spatial = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()  # Gating mechanism for spatial enhancement
        )
        
        # Progressive accumulation weights - models temporal decay
        self.register_buffer('temporal_weights', self._create_temporal_weights(5))
        
    def _create_temporal_weights(self, max_frames):
        """Create temporal decay weights that emphasize recent frames while preserving history"""
        # Exponential decay with stronger emphasis on recent frames
        weights = torch.exp(-0.3 * torch.arange(max_frames, dtype=torch.float32))
        return weights / weights.sum()
    
    def forward(self, frame_features):
        """
        Args:
            frame_features (torch.Tensor): [B, T, feature_dim] - features from all frames
            
        Returns:
            accumulated_features (torch.Tensor): [B, T, feature_dim] - velocity-accumulated features
            velocity_context (torch.Tensor): [B, feature_dim//2] - pure velocity context
        """
        B, T, D = frame_features.shape
        
        # Extract velocity and spatial components from each frame
        velocity_features = []  # Doppler/velocity information
        spatial_features = []   # Spatial structure information
        
        for t in range(T):
            vel_feat = self.velocity_extractor(frame_features[:, t])  # [B, D//2]
            spa_feat = self.spatial_extractor(frame_features[:, t])   # [B, D//2]
            velocity_features.append(vel_feat)
            spatial_features.append(spa_feat)
        
        # Stack features: [B, T, D//2]
        velocity_stack = torch.stack(velocity_features, dim=1)
        spatial_stack = torch.stack(spatial_features, dim=1)
        
        # Apply Doppler attention - velocity features attend to each other
        # This models how velocity information from different frames interact
        attended_velocity, attention_weights = self.doppler_attention(
            velocity_stack, velocity_stack, velocity_stack
        )  # [B, T, D//2]
        
        # Progressive velocity accumulation with temporal weighting
        accumulated_velocity = torch.zeros_like(attended_velocity)
        for t in range(T):
            # Accumulate velocity information from frame 0 to t
            if t == 0:
                accumulated_velocity[:, t] = attended_velocity[:, t]
            else:
                # Weight current frame and accumulated history
                current_weight = self.temporal_weights[0]
                history_weight = self.temporal_weights[1:t+1].sum()
                
                accumulated_velocity[:, t] = (
                    current_weight * attended_velocity[:, t] + 
                    history_weight * accumulated_velocity[:, t-1]
                )
        
        # Transform accumulated velocity to spatial enhancement
        velocity_to_spatial_gates = []
        for t in range(T):
            gate = self.velocity_to_spatial(accumulated_velocity[:, t])  # [B, D]
            velocity_to_spatial_gates.append(gate)
        
        spatial_gates = torch.stack(velocity_to_spatial_gates, dim=1)  # [B, T, D]
        
        # Apply velocity-informed spatial enhancement
        enhanced_features = frame_features * spatial_gates  # Element-wise gating
        
        # Return both enhanced features and pure velocity context for final fusion
        final_velocity_context = accumulated_velocity[:, -1]  # [B, D//2] - final accumulated velocity
        
        return enhanced_features, final_velocity_context


class SpatialVelocityFusion(nn.Module):
    """Fusion module that combines spatial features with accumulated velocity context
    
    Models the physical principle that velocity information directly affects
    spatial feature distribution in mmWave radar data.
    """
    def __init__(self, spatial_dim, velocity_dim):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.velocity_dim = velocity_dim
        
        # Cross-modal attention between spatial and velocity
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=spatial_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Velocity-guided spatial transformation
        self.velocity_guide = nn.Sequential(
            nn.Linear(velocity_dim, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim, spatial_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(spatial_dim * 2, spatial_dim),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim, spatial_dim)
        )
    
    def forward(self, spatial_features, velocity_context):
        """
        Args:
            spatial_features (torch.Tensor): [B, spatial_dim] - spatial features from final frame
            velocity_context (torch.Tensor): [B, velocity_dim] - accumulated velocity context
            
        Returns:
            fused_features (torch.Tensor): [B, spatial_dim] - velocity-enhanced spatial features
        """
        B = spatial_features.shape[0]
        
        # Generate velocity-guided spatial attention
        velocity_guidance = self.velocity_guide(velocity_context)  # [B, spatial_dim]
        
        # Apply velocity guidance to spatial features
        guided_spatial = spatial_features * velocity_guidance  # [B, spatial_dim]
        
        # Cross-modal fusion
        spatial_expanded = spatial_features.unsqueeze(1)  # [B, 1, spatial_dim]
        guided_expanded = guided_spatial.unsqueeze(1)     # [B, 1, spatial_dim]
        
        attended_spatial, _ = self.cross_attention(
            spatial_expanded, guided_expanded, guided_expanded
        )  # [B, 1, spatial_dim]
        attended_spatial = attended_spatial.squeeze(1)  # [B, spatial_dim]
        
        # Final fusion
        concatenated = torch.cat([spatial_features, attended_spatial], dim=1)  # [B, spatial_dim*2]
        fused_features = self.fusion_layer(concatenated)  # [B, spatial_dim]
        
        return fused_features


class CSPRTMTemporalEncoder3D(nn.Module):
    """Enhanced Temporal CSP Encoder with Doppler Velocity Accumulation
    
    Key innovations:
    1. Doppler-specific velocity accumulator that models velocity-to-spatial information flow
    2. Multi-head Doppler attention for velocity component interaction
    3. Progressive velocity accumulation with temporal weighting
    4. Cross-modal spatial-velocity fusion
    
    This design specifically addresses mmWave radar's unique characteristic where
    Doppler (velocity) information from earlier frames directly influences spatial
    feature extraction in later frames.
    """
    def __init__(self, 
                 in_channels=64,
                 base_channels=64,
                 stage_channels=[128, 256, 512, 1024], 
                 stage_blocks=[2, 4, 4, 2],
                 spp_kernel_sizes=(3, 5, 7),
                 expansion=0.5,
                 channel_attention=True,
                 norm_layer='torch.nn.BatchNorm3d',
                 temporal_frames=5,
                 num_velocity_heads=8,
                 spatial_encoder_pretrained=None,
                 spatial_encoder_freeze=False,
                 **kwargs):
        super().__init__()
        
        self.temporal_frames = temporal_frames
        
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
        
        # Doppler velocity accumulator - key innovation
        self.velocity_accumulator = DopplerVelocityAccumulator(
            feature_dim=stage_channels[-1],
            num_velocity_heads=num_velocity_heads
        )
        
        # Spatial-velocity fusion module
        self.spatial_velocity_fusion = SpatialVelocityFusion(
            spatial_dim=stage_channels[-1],
            velocity_dim=stage_channels[-1] // 2
        )
        
        # Enhanced spatial processor with velocity-aware convolutions
        self.spatial_processor = nn.Sequential(
            nn.Conv3d(stage_channels[-1], stage_channels[-1] // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(stage_channels[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(stage_channels[-1] // 2, stage_channels[-1] // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(stage_channels[-1] // 4),
            nn.ReLU(inplace=True)
        )
        
        # Final feature extractor
        self.final_extractor = nn.AdaptiveAvgPool3d(1)
        
        # Feature dimension
        self.feature_dim = stage_channels[-1] // 4
        
        self._init_weights()
    
    def _load_spatial_encoder_pretrained(self, pretrained_path):
        """Load pretrained weights for the spatial encoder
        
        Args:
            pretrained_path (str): Path to the pretrained weights file
        """
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
            
            # Method 1: Direct matching (if the checkpoint is from CSPEncoder3D)
            for key, value in state_dict.items():
                if key.startswith('spatial_encoder.'):
                    # Remove 'spatial_encoder.' prefix
                    new_key = key[len('spatial_encoder.'):]
                    spatial_encoder_state_dict[new_key] = value
                elif not any(prefix in key for prefix in ['velocity_accumulator', 'spatial_velocity_fusion', 'spatial_processor', 'final_extractor']):
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
        """Freeze the spatial encoder parameters to prevent updates during training
        
        This is useful when using pretrained spatial encoder weights and wanting to
        only train the temporal processing components (velocity accumulator and fusion).
        """
        print("Freezing spatial encoder parameters...")
        
        frozen_params = 0
        total_params = 0
        
        for name, param in self.spatial_encoder.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            total_params += param.numel()
        
        print(f"Frozen {frozen_params:,} parameters in spatial encoder")
        print(f"Spatial encoder is now frozen and will not be updated during training")
        
        # Set spatial encoder to evaluation mode to disable batch norm updates
        self.spatial_encoder.eval()
    
    def _unfreeze_spatial_encoder(self):
        """Unfreeze the spatial encoder parameters to allow updates during training
        
        This can be called to enable fine-tuning of the spatial encoder after
        initial training with frozen weights.
        """
        print("Unfreezing spatial encoder parameters...")
        
        unfrozen_params = 0
        
        for name, param in self.spatial_encoder.named_parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
        
        print(f"Unfrozen {unfrozen_params:,} parameters in spatial encoder")
        print(f"Spatial encoder can now be updated during training")
        
        # Set spatial encoder back to training mode
        self.spatial_encoder.train()
    
    def get_trainable_parameters(self):
        """Get information about trainable vs frozen parameters
        
        Returns:
            dict: Dictionary with parameter counts and details
        """
        spatial_trainable = sum(p.numel() for p in self.spatial_encoder.parameters() if p.requires_grad)
        spatial_total = sum(p.numel() for p in self.spatial_encoder.parameters())
        
        velocity_trainable = sum(p.numel() for p in self.velocity_accumulator.parameters() if p.requires_grad)
        velocity_total = sum(p.numel() for p in self.velocity_accumulator.parameters())
        
        fusion_trainable = sum(p.numel() for p in self.spatial_velocity_fusion.parameters() if p.requires_grad)
        fusion_total = sum(p.numel() for p in self.spatial_velocity_fusion.parameters())
        
        processor_trainable = sum(p.numel() for p in self.spatial_processor.parameters() if p.requires_grad)
        processor_total = sum(p.numel() for p in self.spatial_processor.parameters())
        
        total_trainable = spatial_trainable + velocity_trainable + fusion_trainable + processor_trainable
        total_params = spatial_total + velocity_total + fusion_total + processor_total
        
        return {
            'spatial_encoder': {
                'trainable': spatial_trainable,
                'total': spatial_total,
                'frozen': spatial_total - spatial_trainable
            },
            'velocity_accumulator': {
                'trainable': velocity_trainable,
                'total': velocity_total,
                'frozen': velocity_total - velocity_trainable
            },
            'spatial_velocity_fusion': {
                'trainable': fusion_trainable,
                'total': fusion_total,
                'frozen': fusion_total - fusion_trainable
            },
            'spatial_processor': {
                'trainable': processor_trainable,
                'total': processor_total,
                'frozen': processor_total - processor_trainable
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
        if hasattr(self, '_spatial_encoder_frozen') or not any(p.requires_grad for p in self.spatial_encoder.parameters()):
            self.spatial_encoder.eval()
        
        return self
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with Doppler velocity accumulation
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, 64, 32, 32, 32]
        
        Returns:
            global_feat (torch.Tensor): Global feature vector [B, feature_dim]
        """
        B, T, C, H, W, D = x.shape
        
        # Extract spatial features from each frame
        frame_features = []
        for t in range(T):
            frame = x[:, t, :, :, :, :]  # [B, C, H, W, D]
            frame_feat = self.spatial_encoder(frame)  # [B, stage_channels[-1]]
            frame_features.append(frame_feat)
        
        # Stack frame features: [B, T, feature_dim]
        temporal_features = torch.stack(frame_features, dim=1)
        
        # Apply Doppler velocity accumulation - KEY INNOVATION
        enhanced_features, velocity_context = self.velocity_accumulator(temporal_features)
        
        # Use the final frame's enhanced features and accumulated velocity context
        final_spatial_features = enhanced_features[:, -1, :]  # [B, feature_dim]
        
        # Spatial-velocity fusion
        fused_features = self.spatial_velocity_fusion(final_spatial_features, velocity_context)
        
        # Reshape for spatial processing
        fused_expanded = fused_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, feature_dim, 1, 1, 1]
        fused_expanded = fused_expanded.expand(-1, -1, H//4, W//4, D//4)  # [B, feature_dim, H//4, W//4, D//4]
        
        # Process through spatial processor
        processed_frame = self.spatial_processor(fused_expanded)  # [B, feature_dim//4, H//4, W//4, D//4]
        
        # Extract final features
        features = self.final_extractor(processed_frame)  # [B, feature_dim//4, 1, 1, 1]
        global_feat = features.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, feature_dim//4]
        
        return global_feat