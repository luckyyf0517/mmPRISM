import torch
import torch.nn as nn
from src.model.stgcn_layers import Graph, get_stgcn_chain

class HandPoseEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Initialize graphs for body and hands
        self.modes = ['body', 'left', 'right']
        self.graph = {}
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        
        # Projection layer
        self.proj_linear = nn.Linear(3, hidden_dim)
        
        # Create graph and GCN for body and hands
        for mode in self.modes:
            if mode == 'body':
                self.graph[mode] = Graph(layout='body', strategy='distance', max_hop=1)
            else:
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

    def forward(self, x):
        """
        Input: x [B, N, 2, 24, 3] - batch, frames, hands(left/right), joints, coords
        Output: [B, N, C] - C is the final feature dimension
        """
        features = []
        
        # Reshape input data format
        x = {
            'body': torch.cat([x[:, :, 0, :3], x[:, :, 1, :3]], dim=2),  # Concatenate the first 3 points of both hands
            'left': x[:, :, 0, 3:],  # All points of the left hand
            'right': x[:, :, 1, 3:]  # All points of the right hand
        }
        
        # Process body features first
        body_data = x['body']  # [B, N, 6, 3]
        body_proj = self.proj_linear(body_data)
        body_proj = body_proj.permute(0, 3, 1, 2)  # [B, C, N, 6]
        body_feat = self.gcn_modules['body'](body_proj)
        body_feat = self.fusion_gcn_modules['body'](body_feat)
        # Add body features to output
        pool_body_feat = body_feat.mean(-1).transpose(1, 2)  # [B, N, C]
        features.append(pool_body_feat)
        # Process left and right hands
        for mode in ['left', 'right']:
            # Get data for one hand [B, N, 24, 3]
            hand_data = x[mode]
            # Project to hidden dim [B, N, 24, hidden_dim]
            proj_feat = self.proj_linear(hand_data)
            proj_feat = proj_feat.permute(0, 3, 1, 2)
            # Forward pass through spatial GCN
            spatial_feat = self.gcn_modules[mode](proj_feat)
            # Add body reference features
            if mode == 'left':
                ref_feat = body_feat[..., [2]]
            else:
                ref_feat = body_feat[..., [5]]
            spatial_feat = spatial_feat + ref_feat.detach()
            # Forward pass through temporal GCN
            temporal_feat = self.fusion_gcn_modules[mode](spatial_feat)
            # Average pooling over node dimension [B, C, N]
            pool_feat = temporal_feat.mean(-1)
            # Rearrange dimensions to [B, N, C]
            pool_feat = pool_feat.transpose(1, 2)
            features.append(pool_feat)
        
        # Merge features from both hands
        output = torch.cat(features, dim=-1)  # [B, N, C*2]
        return output
