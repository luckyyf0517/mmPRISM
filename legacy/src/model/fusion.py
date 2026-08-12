import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class CrossAttentionFusion(nn.Module):
    """Advanced Cross-Attention based multimodal fusion module"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Pose as primary modality - define as query
        self.pose_to_q = nn.Linear(dim, dim, bias=False)
        
        # Features as auxiliary modality - define as key and value
        self.feature_to_k = nn.Linear(dim, dim, bias=False)
        self.feature_to_v = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization - separate for each modality
        self.pose_norm = nn.LayerNorm(dim)
        self.feature_norm = nn.LayerNorm(dim)
        
    def forward(self, pose_embeds, feature_embeds):
        """
        Args:
            pose_embeds: [B, T, D] - Primary modality (pose information)
            feature_embeds: [B, T, D] - Auxiliary modality (feature information)
        Returns:
            fused_embeds: [B, T, D] - Fused embeddings with pose as primary
        """
        B, T, D = pose_embeds.shape
        
        # Validate input dimensions
        if feature_embeds.shape != pose_embeds.shape:
            raise ValueError(f"Dimension mismatch: pose_embeds {pose_embeds.shape} vs feature_embeds {feature_embeds.shape}")
        
        # Apply separate layer normalization for each modality
        pose_embeds = self.pose_norm(pose_embeds)
        feature_embeds = self.feature_norm(feature_embeds)
        
        # Generate Q from pose (primary), K,V from features (auxiliary)
        q = self.pose_to_q(pose_embeds)  # [B, T, D]
        k = self.feature_to_k(feature_embeds)  # [B, T, D]
        v = self.feature_to_v(feature_embeds)  # [B, T, D]
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.num_heads)
        
        # Compute attention scores
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        # Output projection
        out = self.proj(out)
        out = self.dropout(out)
        
        # Residual connection with pose as primary
        fused_embeds = pose_embeds + out
        
        return fused_embeds


class TemporalAlignmentModule(nn.Module):
    """Temporal alignment module for pose and feature modalities"""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Temporal attention for alignment
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Temporal convolution for local alignment
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, pose_embeds, feature_embeds):
        """
        Align pose and feature modalities in temporal dimension
        """
        B, T, D = pose_embeds.shape
        
        # Cross-temporal attention between modalities
        aligned_features, _ = self.temporal_attention(
            query=pose_embeds,
            key=feature_embeds,
            value=feature_embeds
        )
        
        # Local temporal convolution for fine-grained alignment
        aligned_features = aligned_features.transpose(1, 2)  # [B, D, T]
        aligned_features = self.temporal_conv(aligned_features)
        aligned_features = aligned_features.transpose(1, 2)  # [B, T, D]
        
        # Residual connection
        aligned_features = self.norm(aligned_features + feature_embeds)
        
        return aligned_features


class AdaptiveFusionGate(nn.Module):
    """Learnable fusion gate for adaptive modality combination"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Cross-modal interaction
        self.cross_interaction = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # Modality-specific gates
        self.pose_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.feature_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, pose_embeds, feature_embeds):
        """
        Generate adaptive fusion gates
        """
        B, T, D = pose_embeds.shape
        
        # Cross-modal interaction for gate generation
        cross_input = torch.cat([pose_embeds, feature_embeds], dim=-1)
        cross_gate = self.cross_interaction(cross_input)
        
        # Modality-specific gates
        pose_gate = self.pose_gate(pose_embeds)
        feature_gate = self.feature_gate(feature_embeds)
        
        # Combine gates
        final_pose_gate = pose_gate * cross_gate
        final_feature_gate = feature_gate * (1 - cross_gate)
        
        return final_pose_gate, final_feature_gate


class EnhancedDynamicModalityFusion(nn.Module):
    """Enhanced dynamic modality fusion with advanced techniques"""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Temporal alignment module
        self.temporal_alignment = TemporalAlignmentModule(dim)
        
        # Cross-Attention fusion
        self.cross_attention = CrossAttentionFusion(dim, dropout=dropout)
        
        # Adaptive fusion gate
        self.fusion_gate = AdaptiveFusionGate(dim)
        
        # Learnable cross-attention weight (instead of hard-coded 0.1)
        self.ca_weight = nn.Parameter(torch.tensor(0.1))
        
        # Feature refinement with better architecture
        self.feature_refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Multi-scale temporal modeling
        self.temporal_scales = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, pose_embeds, feature_embeds):
        """
        Args:
            pose_embeds: [B, T, D] - Primary modality (pose information)
            feature_embeds: [B, T, D] - Auxiliary modality (feature information)
        Returns:
            fused_embeds: [B, T, D] - Enhanced dynamically fused embeddings
        """
        B, T, D = pose_embeds.shape
        
        # Validate input dimensions
        if feature_embeds.shape != pose_embeds.shape:
            raise ValueError(f"Dimension mismatch: pose_embeds {pose_embeds.shape} vs feature_embeds {feature_embeds.shape}")
        
        # Step 1: Temporal alignment
        aligned_features = self.temporal_alignment(pose_embeds, feature_embeds)
        
        # Step 2: Feature refinement
        refined_features = self.feature_refine(aligned_features)
        
        # Step 3: Multi-scale temporal modeling for features
        multi_scale_features = []
        for conv in self.temporal_scales:
            temp_feat = conv(refined_features.transpose(1, 2)).transpose(1, 2)
            multi_scale_features.append(temp_feat)
        
        # Combine multi-scale features
        multi_scale_features = torch.stack(multi_scale_features, dim=1).mean(dim=1)
        refined_features = refined_features + 0.1 * multi_scale_features
        
        # Step 4: Cross-attention fusion with pose as primary
        ca_fused = self.cross_attention(pose_embeds, refined_features)
        
        # Step 5: Adaptive fusion gates
        pose_gate, feature_gate = self.fusion_gate(pose_embeds, refined_features)
        
        # Step 6: Gated combination
        gated_pose = pose_gate * pose_embeds
        gated_features = feature_gate * refined_features
        
        # Step 7: Final fusion with learnable cross-attention weight
        dynamically_fused = gated_pose + gated_features
        final_fused = dynamically_fused + torch.sigmoid(self.ca_weight) * ca_fused
        
        # Step 8: Final refinement
        final_fused = self.final_fusion(
            torch.cat([final_fused, pose_embeds], dim=-1)
        )
        
        return final_fused


class DynamicModalityFusion(nn.Module):
    """Dynamic modality fusion with learnable weighting"""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Modality importance predictor
        self.modality_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Cross-Attention fusion
        self.cross_attention = CrossAttentionFusion(dim, dropout=dropout)
        
        # Feature refinement for auxiliary modality
        self.feature_refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, pose_embeds, feature_embeds):
        """
        Args:
            pose_embeds: [B, T, D] - Primary modality (pose information)
            feature_embeds: [B, T, D] - Auxiliary modality (feature information)
        Returns:
            fused_embeds: [B, T, D] - Dynamically fused embeddings
        """
        B, T, D = pose_embeds.shape
        
        # Validate input dimensions
        if feature_embeds.shape != pose_embeds.shape:
            raise ValueError(f"Dimension mismatch: pose_embeds {pose_embeds.shape} vs feature_embeds {feature_embeds.shape}")
        
        # Refine auxiliary features
        refined_features = self.feature_refine(feature_embeds)
        
        # Cross-attention fusion with pose as primary
        ca_fused = self.cross_attention(pose_embeds, refined_features)
        
        # Global pooling for modality importance prediction
        pose_pool = pose_embeds.mean(dim=1)  # [B, D]
        feature_pool = refined_features.mean(dim=1)  # [B, D]
        
        # Concatenate global features
        global_features = torch.cat([pose_pool, feature_pool], dim=-1)  # [B, 2D]
        
        # Predict modality importance
        weights = self.modality_predictor(global_features)  # [B, 2]
        
        # Expand weights for sequence-level application
        weights = repeat(weights, 'b m -> b t m', t=T)  # [B, T, 2]
        
        # Apply dynamic weighting
        pose_weight = weights[:, :, 0:1]  # [B, T, 1]
        feature_weight = weights[:, :, 1:2]  # [B, T, 1]
        
        # Weighted combination
        dynamically_fused = pose_weight * pose_embeds + feature_weight * refined_features
        
        # Combine with cross-attention result
        # Use cross-attention as residual enhancement
        final_fused = dynamically_fused + 0.1 * ca_fused
        
        return final_fused