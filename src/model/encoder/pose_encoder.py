import torch
import torch.nn as nn

import sys
sys.path.append('.')

from src.model.stgcn_layers import Graph, get_stgcn_chain

class HandPoseEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=768):
        super().__init__()
        
        # Initialize graphs for body and hands
        self.modes = ['body', 'hand']
        self.graph = {}
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        
        # Create graph and GCN for body and hand
        for mode in self.modes:
            if mode == 'body':
                self.graph[mode] = Graph(layout='body', strategy='distance', max_hop=1)
            else:  # hand
                self.graph[mode] = Graph(layout='hand', strategy='distance', max_hop=1)
            
            A = torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False)
            
            # Create spatial and temporal GCN modules
            spatial_kernel_size = A.size(0)
            self.gcn_modules[mode], final_dim = get_stgcn_chain(
                hidden_dim, 
                'spatial', 
                (1, spatial_kernel_size), 
                A.clone(), 
                True
            )
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(
                final_dim,
                'temporal',
                (5, spatial_kernel_size),
                A.clone(),
                True
            )

        # Projection layer for input coordinates
        self.proj_linear = nn.Linear(input_dim, hidden_dim)
        
        # Learnable parameters for part importance
        self.part_para = nn.Parameter(torch.zeros(final_dim * 3), requires_grad=True)  # For body, left, right
        
        # Final projection to LLM hidden dimension
        self.final_projection = nn.Linear(final_dim * 3, output_dim)  # For body, left, right

    def forward(self, x):
        """
        Process pose data through GCN networks
        
        Args:
            x: Input pose data [B, N, 2, 24, 3] - batch, frames, hands(left/right), joints, coords
        
        Returns:
            Feature tensor [B, N, C] aligned with LLM hidden dimension
        """
        # Prepare input data for each part
        parts_data = {
            'body': torch.cat([
                (x[:, :, 0, 0] + x[:, :, 1, 0]).unsqueeze(-2) / 2,  # Average of first points
                x[:, :, 0, :3],  # First 3 points from left
                x[:, :, 1, :3],  # First 3 points from right
            ], dim=-2),  # Total 7 points: 1 avg + 3 left + 3 right
            'left': x[:, :, 0, 3:],  # Left hand points
            'right': x[:, :, 1, 3:]  # Right hand points
        }
        
        # Define which GCN module to use for each part
        part_to_mode = {
            'body': 'body',
            'left': 'hand',
            'right': 'hand'
        }
        
        # Reference points from body to hands
        ref_points = {
            'left': 3,  # Index of reference point for left hand
            'right': 6  # Index of reference point for right hand
        }
        
        features = []
        spatial_features = {}
        
        # Process all parts in a single loop
        for part in ['body', 'left', 'right']:
            # Get data for current part
            part_data = parts_data[part]
            
            # Project to hidden dim
            proj_feat = self.proj_linear(part_data).permute(0, 3, 1, 2)  # [B, C, N, V]
            
            # Get the appropriate GCN module
            mode = part_to_mode[part]
            
            # Forward pass through spatial GCN
            spatial_feat = self.gcn_modules[mode](proj_feat)
            
            # Store spatial features for reference
            spatial_features[part] = spatial_feat
            
            # Add body reference features for hands
            if part != 'body':
                ref_idx = ref_points[part]
                ref_feat = spatial_features['body'][..., [ref_idx]]
                spatial_feat = spatial_feat + ref_feat.detach()
            
            # Forward pass through temporal GCN
            temporal_feat = self.fusion_gcn_modules[mode](spatial_feat)
            
            # Average pooling over node dimension and rearrange to [B, N, C]
            pool_feat = temporal_feat.mean(-1).transpose(1, 2)
            features.append(pool_feat)
        
        # Merge features from all parts
        combined_features = torch.cat(features, dim=-1)
        combined_features = combined_features + self.part_para
        
        # Project to LLM hidden dimension
        output = self.final_projection(combined_features)  # [B, N, output_dim]
        
        return output


class StaticPoseEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=768):
        super().__init__()
        
        # Initialize graph for hand
        self.graph = Graph(layout='hand', strategy='distance', max_hop=1)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        
        # Create spatial GCN modules
        spatial_kernel_size = A.size(0)
        self.gcn_module, final_dim = get_stgcn_chain(
            hidden_dim, 
            'spatial', 
            (1, spatial_kernel_size), 
            A.clone(), 
            True
        )

        # Projection layer for input coordinates
        self.proj_linear = nn.Linear(input_dim, hidden_dim)
        
        # Final projection to output dimension
        self.final_projection = nn.Linear(final_dim, output_dim)
    
    def forward(self, x):
        """
        Process single hand pose data
        
        Args:
            x: Input pose data [B, 21, 3] - batch, joints, coords
        
        Returns:
            Feature tensor [B, C] aligned with output dimension
        """
        # Project to hidden dim and add time dimension
        proj_feat = self.proj_linear(x).permute(0, 2, 1)  # [B, C, V]
        proj_feat = proj_feat.unsqueeze(2)  # [B, C, 1, V] - add time dimension
        
        # Forward pass through spatial GCN
        spatial_feat = self.gcn_module(proj_feat)
        
        # Average pooling over node dimension and rearrange to [B, C]
        pool_feat = spatial_feat.mean(-1).squeeze(2)  # Remove time dimension and pool over nodes
        
        # Project to output dimension
        output = self.final_projection(pool_feat)  # [B, output_dim]
        
        return output
