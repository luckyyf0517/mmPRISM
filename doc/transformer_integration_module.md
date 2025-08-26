# Transformer Integration Module in OmniHand

## Overview

The transformer integration module in `src/model/encoder/cubenet_rtm_transformer.py` implements an advanced temporal processing architecture for 3D hand pose estimation from millimeter wave radar data. This module enhances the base CSP-RTM encoder with state-of-the-art transformer-based temporal modeling capabilities.

## Key Components

### 1. RoPE Positional Encoding (RoPEPositionalEncoding)
Implements Rotary Position Embedding for improved temporal modeling:
- Provides better relative position awareness
- Generalizes better to sequences longer than those seen during training
- Uses frequency-based rotary transformations to encode temporal positions

### 2. Temporal Convolution Block (TemporalConvBlock)
Captures short-term temporal dependencies:
- 1D temporal convolution with dilated convolutions
- Depthwise-separable convolution for efficiency
- Layer normalization and residual connections

### 3. SwiGLU Feedforward Network (SwiGLUFeedForward)
Implements a gating mechanism for feature transformation:
- Uses SiLU activation with gating (SwiGLU)
- Provides better performance than traditional ReLU-based feedforward networks
- Efficient feature transformation with gating mechanism

### 4. Enhanced Transformer Layer (EnhancedTransformerLayer)
Core building block combining multiple advanced techniques:
- Multi-head self-attention with RoPE encoding
- Temporal convolution branch for local dependencies
- SwiGLU feedforward networks
- Gated fusion of conv and attention features
- Pre-normalization architecture

### 5. Advanced Temporal Transformer Encoder (AdvancedTemporalTransformerEncoder)
High-level temporal aggregation module:
- CLS token with learnable temperature for global representation
- Multiple aggregation strategies (CLS, mean pooling, attention pooling)
- Cross-attention for spatial-temporal feature fusion
- Learnable weights for aggregation strategy fusion

### 6. CSP-RTM Transformer Encoder 3D (CSPRTMTransformerEncoder3D)
Main encoder module integrating all components:
- Shared CSPEncoder3D for spatial feature extraction
- Advanced temporal transformer for temporal aggregation
- Enhanced output projection with gating mechanisms
- Support for pretrained spatial encoder loading and freezing

## Architecture Details

### Data Flow
1. **Input**: Temporal sequence of mmwave cubes [B, T, 64, 32, 32, 32]
2. **Spatial Encoding**: Shared CSPEncoder3D processes each frame independently
3. **Temporal Processing**: AdvancedTransformerEncoder aggregates temporal information
4. **Output**: Global feature vector [B, output_dim]

### Temporal Processing Pipeline
1. **Feature Extraction**: Each frame processed through shared spatial encoder
2. **Positional Encoding**: RoPE applied to transformer attention mechanisms
3. **Local Temporal Modeling**: Temporal convolutions capture short-term dependencies
4. **Global Temporal Modeling**: Multi-head attention captures long-term dependencies
5. **Feature Fusion**: Gated combination of convolutional and attention features
6. **Aggregation**: Multiple strategies for temporal sequence summarization

### Key Innovations
1. **RoPE Integration**: Better relative position modeling in temporal sequences
2. **Gated Feature Fusion**: Controlled combination of local and global temporal features
3. **Multiple Aggregation Strategies**: Robust temporal feature summarization
4. **Cross-Attention Fusion**: Enhanced interaction between spatial and temporal features
5. **SwiGLU Feedforward**: Improved feature transformation quality

## Configuration

The module is configured through the YAML configuration file:

```yaml
backbone: 
  target: src.model.encoder.cubenet_rtm_transformer.CSPRTMTransformerEncoder3D
  params: 
    # Spatial encoder parameters
    in_channels: 64
    base_channels: 64
    stage_channels: [128, 256, 512, 1024]
    stage_blocks: [2, 4, 4, 2]
    spp_kernel_sizes: [3, 5, 7]
    expansion: 0.5
    norm_layer: torch.nn.GroupNorm
    
    # Pretrained spatial encoder settings
    spatial_encoder_pretrained: weights/rtm_0824/model.pth
    spatial_encoder_freeze: false
    
    # Enhanced temporal transformer parameters
    temporal_frames: 8
    transformer_layers: 8
    transformer_heads: 16
    transformer_dim_feedforward: 3072
    transformer_dropout: 0.1
    
    # Advanced temporal modeling features
    temporal_kernel_size: 3
    use_cross_attention: true
    
    # Output parameters
    output_dim: 512
```

## Training

The model is trained using the script:
```bash
torchrun --nproc_per_node=2 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand/omnihand_rtm_collected_transformer.yaml \
    --batch-size 8 \
    --max-epochs 30 \
    --version "omnihand_rtm_collected_transformer_exp" \
    --precision 32 \
    --seed 42
```

## Advantages

1. **Enhanced Temporal Modeling**: Combines local convolutions with global attention
2. **Improved Positional Awareness**: RoPE provides better sequence modeling
3. **Robust Feature Aggregation**: Multiple strategies ensure reliable temporal summarization
4. **Scalable Architecture**: Configurable layers, heads, and dimensions
5. **Efficient Processing**: Shared spatial encoder reduces computational cost