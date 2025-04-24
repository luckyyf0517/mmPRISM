import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder for millimeter wave signals"""
    
    def __init__(self, input_channels, output_channels, hidden_dims, kernel_sizes, strides, paddings, use_batch_norm=True, use_residual=True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Ensure all lists have the same length
        assert len(hidden_dims) == len(kernel_sizes) == len(strides) == len(paddings), \
            "All parameter lists must have the same length"
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(len(hidden_dims)):
            out_channels = hidden_dims[i]
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            
            # Create layer
            layer = nn.Sequential()
            
            # Add convolution
            layer.add_module(f"conv{i}", nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=not use_batch_norm
            ))
            
            # Add batch norm if requested
            if use_batch_norm:
                layer.add_module(f"bn{i}", nn.BatchNorm1d(out_channels))
            
            # Add activation
            layer.add_module(f"relu{i}", nn.ReLU(inplace=True))
            
            # Add layer to module list
            self.layers.append(layer)
            
            # Update in_channels for next layer
            in_channels = out_channels
        
        # Final projection layer to match model hidden size
        self.final_proj = nn.Conv1d(hidden_dims[-1], output_channels, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [B, T, C, H, W] or [B, T, C]
            
        Returns:
            Tensor of shape [B, T, output_channels]
        """
        # Handle different input shapes
        if len(x.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = x.shape
            # Reshape to [B*T, C, H*W]
            x = x.view(B*T, C, H*W)
        elif len(x.shape) == 3:  # [B, T, C]
            B, T, C = x.shape
            # Reshape to [B*T, C, 1]
            x = x.view(B*T, C, 1)
        else:
            raise ValueError(f"Expected input shape [B, T, C, H, W] or [B, T, C], got {x.shape}")
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final projection
        x = self.final_proj(x)
        
        # Reshape back to [B, T, output_channels]
        if len(x.shape) == 3:  # [B*T, output_channels, L]
            x = x.mean(dim=2)  # Global average pooling
            x = x.view(B, T, self.output_channels)
        
        return x 