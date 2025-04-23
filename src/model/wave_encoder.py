import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer

class WaveEncoder(nn.Module):
    """Wave signal encoder using Vision Transformer
    
    Args:
        img_size (tuple): Input image size (H, W)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer layers
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dimension ratio
        drop_rate (float): Dropout rate
    """
    def __init__(self, 
                 img_size=(64, 64),
                 patch_size=8,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 drop_rate=0.1):
        super().__init__()
        
        # Initialize Vision Transformer
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            num_classes=0  # Set to 0 to get features instead of classification
        )

    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Output features of shape [B, embed_dim]
        """
        features = self.vit(x)
        return features


if __name__ == "__main__":
    # Test WaveEncoder
    print("Testing WaveEncoder...")
    
    # Create encoder
    encoder = WaveEncoder(
        img_size=(64, 32),
        patch_size=8,
        in_channels=2,
        embed_dim=768
    )
    
    # Create dummy input [B, C, H, W]
    batch_size = 2
    x = torch.randn(batch_size, 2, 64, 32)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    features = encoder(x)
    print(f"Output shape: {features.shape}")
    
    # Check output shape
    expected_shape = (batch_size, 768)
    assert features.shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, got {features.shape}"
    
    print("Test passed successfully!")