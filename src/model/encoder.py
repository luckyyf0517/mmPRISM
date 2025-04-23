import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed
from torchvision import models


class TemporalImageEncoder(nn.Module):
    def __init__(
        self,
        resnet_version='resnet50',  # options: 'resnet18', 'resnet34', 'resnet50', 'resnet101'
        pretrained=True,            # whether to use pretrained weights
        input_channels=2,          # input channels
        output_channels=256,        # output feature dimension
        dropout=0.1                 # dropout rate
    ):
        super().__init__()
        
        # Load pretrained ResNet
        if resnet_version == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif resnet_version == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        elif resnet_version == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif resnet_version == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get ResNet output channels
        if resnet_version in ['resnet18', 'resnet34']:
            backbone_channels = 512
        else:  # resnet50, resnet101
            backbone_channels = 2048
            
        # Add a 1x1 conv layer to adjust channels
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(backbone_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x):
        # Input: [B, N, C, H, W]
        B, N, C, H, W = x.shape
        
        # Reshape input to process all frames at once [B*N, C, H, W]
        x = x.view(B*N, C, H, W)
        
        # ResNet feature extraction for all frames
        feat = self.backbone(x)  # [B*N, backbone_channels, H/32, W/32]
        
        # Adjust channels
        feat = self.channel_adjust(feat)  # [B*N, output_channels, H/32, W/32]
        
        # Global average pooling
        feat = feat.squeeze(-1).squeeze(-1)  # [B*N, output_channels]
        
        # Reshape back to [B, N, output_channels]
        features = feat.view(B, N, -1)
        return features
    
    
if __name__ == '__main__':
    # Create model
    model = TemporalImageEncoder(
        resnet_version='resnet50',  # Use ResNet50
        pretrained=True,            # Use pretrained weights
        output_channels=256,        # Output 256-dim features
        dropout=0.1                 # Dropout rate
    )
    
    # Create example input
    batch_size = 2
    num_frames = 8
    channels = 3
    height = 224
    width = 224
    
    # Create random input tensor
    x = torch.randn(batch_size, num_frames, channels, height, width)
    
    # Move model and input to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Print input/output shapes
    print('='*50)
    print('Input shape:', x.shape)
    print('Output shape:', output.shape)
    print('='*50)
    print('Model test passed!')