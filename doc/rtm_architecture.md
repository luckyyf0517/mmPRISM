# RTM Architecture Documentation

## Overview

The RTM (Radar Task Model) architecture is a family of 3D CNN-based encoders designed for processing millimeter wave (mmWave) radar data for hand pose estimation. The architecture includes several variants that handle both single-frame and temporal sequence processing:

1. **Base RTM (cubenet_rtm.py)**: Core 3D CNN encoder with CSPNeXt blocks
2. **Temporal RTM (cubenet_rtm_temporal.py)**: Extension with Doppler velocity accumulation
3. **Transformer RTM (cubenet_rtm_transformer.py)**: Temporal processing using transformer attention
4. **LSTM RTM (cubenet_rtm_lstm.py)**: Temporal processing using LSTM with attention

## Base RTM Architecture (cubenet_rtm.py)

The base RTM encoder implements a CSPNeXt-based 3D CNN architecture with several key components:

### Key Components

1. **Stem Layer**: Initial feature extraction with downsampling
2. **CSP Stages**: Four stages of Cross-Stage Partial blocks with:
   - CSPNeXt blocks for feature extraction
   - Channel attention modules (optional)
   - Progressive downsampling
3. **SPP Module**: Spatial Pyramid Pooling for multi-scale feature extraction
4. **PAFPN**: Path Aggregation Feature Pyramid Network for feature fusion
5. **Global Pooling**: Final feature vector extraction

### Design Features

- **Component Switches**: Ablation study support with configurable components:
  - `use_csp`: Enable/disable Cross-Stage Partial structure
  - `use_channel_attention`: Enable/disable Channel Attention modules
  - `use_pafpn`: Enable/disable Path Aggregation Feature Pyramid Network
- **Modular Design**: Each component can be independently enabled/disabled
- **Scalable Architecture**: Configurable channel sizes and block counts

### Key Classes

1. **CSPEncoder3D**: Main encoder class with configurable components
2. **CSPBlock3D**: Cross Stage Partial block implementation
3. **CSPNeXtBlock3D**: CSPNeXt bottleneck block
4. **SPP3D**: Spatial Pyramid Pooling for 3D tensors
5. **CSPPAFPN3D**: CSPNeXt Path Aggregation Feature Pyramid Network

## Temporal RTM Architecture (cubenet_rtm_temporal.py)

The temporal RTM extends the base architecture with Doppler velocity accumulation, specifically designed for mmWave radar sequences where velocity information influences spatial features over time.

### Key Innovations

1. **Doppler Velocity Accumulator**: Models how Doppler (velocity) information from earlier frames accumulates and influences spatial feature extraction in later frames
2. **Multi-head Doppler Attention**: Models interactions between different velocity components
3. **Progressive Velocity Accumulation**: Temporal weighting that emphasizes recent frames while preserving history
4. **Spatial-Velocity Fusion**: Combines spatial features with accumulated velocity context

### Key Components

1. **DopplerVelocityAccumulator**: Extracts and accumulates velocity information across frames
2. **SpatialVelocityFusion**: Fuses spatial features with accumulated velocity context
3. **CSPRTMTemporalEncoder3D**: Main encoder class integrating temporal processing

### Architecture Flow

1. Shared spatial encoder processes each frame independently
2. Doppler velocity accumulator models velocity-to-spatial information flow
3. Spatial-velocity fusion combines enhanced spatial features with accumulated velocity context
4. Final feature extraction through spatial processing and global pooling

## Transformer RTM Architecture (cubenet_rtm_transformer.py)

The transformer RTM uses transformer-based attention mechanisms for temporal sequence processing.

### Key Components

1. **PositionalEncoding**: Learnable positional embeddings for temporal order encoding
2. **EnhancedTemporalTransformerEncoder**: Transformer encoder with CLS token for temporal aggregation
3. **CSPRTMTransformerEncoder3D**: Main encoder class integrating transformer temporal processing

### Design Features

- **CLS Token**: Learnable global aggregation token for sequence representation
- **Learnable Positional Encoding**: Better initialization for temporal order representation
- **Pre-norm Transformer**: Improved training stability with normalization before attention/feedforward
- **Residual Connections**: Enhanced gradient flow in deep networks

### Architecture Flow

1. Shared spatial encoder processes each frame independently
2. Positional encoding added to frame features
3. Transformer encoder with CLS token aggregates temporal information
4. CLS token output used as final aggregated representation
5. Output projection generates final feature vector

## LSTM RTM Architecture (cubenet_rtm_lstm.py)

The LSTM RTM uses bidirectional LSTM with attention mechanisms for temporal processing.

### Key Components

1. **TemporalLSTMEncoder**: Bidirectional LSTM with attention mechanism
2. **CSPRTMLSTMEncoder3D**: Main encoder class integrating LSTM temporal processing

### Design Features

- **Bidirectional LSTM**: Captures both past and future context
- **Attention Mechanism**: Weights important temporal frames
- **LSTM for Sequential Processing**: Efficient for modeling long-range temporal dependencies
- **Feature Projection**: Maintains consistent feature dimensions

### Architecture Flow

1. Shared spatial encoder processes each frame independently
2. Bidirectional LSTM processes temporal sequence
3. Attention mechanism weights important frames
4. Weighted LSTM outputs aggregated for final representation
5. Output projection generates final feature vector

## Comparison of Temporal Variants

| Feature | Temporal RTM | Transformer RTM | LSTM RTM |
|---------|--------------|-----------------|----------|
| Temporal Modeling | Doppler velocity accumulation | Self-attention with CLS token | Bidirectional LSTM with attention |
| Computational Complexity | Medium | High | Low-Medium |
| Long-range Dependencies | Good | Excellent | Good |
| Physical Principles | Explicitly models mmWave physics | General attention mechanism | Sequential processing |
| Training Stability | Good | Improved with pre-norm | Generally stable |
| Key Innovation | Velocity-to-spatial information flow | Global context aggregation | Sequential context modeling |

## Usage Examples

### Base RTM
```python
encoder = CSPEncoder3D(
    in_channels=64,
    base_channels=64,
    stage_channels=[128, 256, 512, 1024],
    use_csp=True,
    use_channel_attention=True,
    use_pafpn=True
)
```

### Temporal RTM
```python
encoder = CSPRTMTemporalEncoder3D(
    in_channels=64,
    temporal_frames=5,
    num_velocity_heads=8
)
```

### Transformer RTM
```python
encoder = CSPRTMTransformerEncoder3D(
    in_channels=64,
    temporal_frames=5,
    transformer_layers=6,
    transformer_heads=8
)
```

### LSTM RTM
```python
encoder = CSPRTMLSTMEncoder3D(
    in_channels=64,
    temporal_frames=5,
    lstm_hidden_dim=512,
    lstm_num_layers=2
)
```