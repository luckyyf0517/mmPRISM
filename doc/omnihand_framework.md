# OmniHand Framework Documentation

## Overview

OmniHand is a deep learning framework for 3D hand pose estimation from millimeter wave (mmWave) radar data. The system processes raw mmWave signals to reconstruct 3D hand poses with high accuracy, enabling gesture recognition and hand tracking applications in challenging environments where optical cameras may fail.

## System Architecture

### Core Components

1. **Data Processing Pipeline**
   - Raw mmWave radar signal acquisition
   - Signal processing (Range FFT, Doppler FFT, Beamforming)
   - Feature extraction from processed radar cubes

2. **Neural Network Architecture**
   - 3D CNN backbone for spatial feature extraction
   - Temporal processing modules for sequence modeling
   - Pose regression head for 3D coordinate prediction

3. **Training Framework**
   - PyTorch Lightning for distributed training
   - Loss functions for pose accuracy optimization
   - Evaluation metrics for performance assessment

## Data Flow

### 1. Input Data
The system accepts two types of input data:
- **Simulated Data**: 3D joint coordinates and velocities for synthetic radar signal generation
- **Real Data**: Raw mmWave radar signals in complex format [num_chirps, num_antenna, num_samples]

### 2. Signal Processing
The `Processor` module converts raw radar signals into 5D mmWave cubes:
```
Input:  [B, num_chirps, num_antenna, num_samples] (complex)
Output: [B, 64, 32, 32, 32] (Doppler, Range, Width, Height)
```

Processing steps:
1. **Range FFT**: Converts time-domain samples to range-domain
2. **Doppler FFT**: Computes velocity information from chirp sequences
3. **Beamforming**: Spatial filtering to create 3D point clouds

### 3. Feature Extraction
The backbone encoder processes mmWave cubes:
- **Base Encoder**: 3D CNN with CSPNeXt blocks
- **Temporal Encoder**: Processes sequences of mmWave cubes for improved accuracy

### 4. Pose Regression
The regressor maps extracted features to 3D joint coordinates:
- Output shape: [B, 2, 24, 3] (batch, left/right hand, joints, xyz coordinates)
- 24 joints per hand: 3 body joints + 21 finger joints

## Model Architecture

### OmniHand Class
The main model class `OmniHand` inherits from PyTorch Lightning's `LightningModule` and includes:

#### Key Components:
1. **Simulator**: Generates synthetic radar signals from 3D poses (training only)
2. **Processor**: Converts raw radar signals to mmWave cubes
3. **Backbone**: 3D CNN encoder for feature extraction
4. **Regressor**: Maps features to 3D joint coordinates
5. **Error Regressor**: Predicts per-joint error estimates
6. **Discriminator**: Optional GAN component for pose realism

#### Forward Pass:
```python
def forward(self, input_data):
    features = self.encode_feature(input_data)
    regressor_output = self.forward_feature(features)
    return {
        'joints': regressor_output[..., :3],  # 3D coordinates
        'error': regressor_output[..., -1],   # Error estimates
    }
```

### Backbone Encoders

#### 1. Base Encoder (CSPEncoder3D)
- CSPNeXt-based 3D CNN architecture
- Configurable stages with increasing channels
- Spatial Pyramid Pooling for multi-scale features
- Path Aggregation Feature Pyramid Network (optional)

#### 2. Temporal Encoder Variants
Three temporal processing approaches:

##### a) Doppler Velocity Accumulation (TVAN-inspired)
Key innovations:
- Explicit modeling of Doppler-to-spatial information flow
- Multi-head Doppler attention for velocity component interaction
- Progressive velocity accumulation with temporal weighting
- Cross-modal spatial-velocity fusion

##### b) Transformer-based Temporal Processing
- Self-attention mechanism with CLS token
- Learnable positional encoding
- Pre-normalization for training stability

##### c) LSTM-based Temporal Processing
- Bidirectional LSTM with attention mechanism
- Computationally efficient for short sequences

## Training Process

### Configuration
Training is configured through YAML files that specify:
- Data paths and preprocessing options
- Model architecture parameters
- Training hyperparameters
- Loss function weights

### Data Loading
The `CollectedSingleFrameDataset` handles:
- Loading 3D joint ground truth data
- Loading corresponding mmWave radar signals
- Temporal sequence preparation for temporal models
- Data normalization and augmentation

### Loss Functions
#### Primary Losses:
1. **L1 Loss**: Direct coordinate error minimization
2. **MPJPE**: Mean Per Joint Position Error
3. **Error Regression Loss**: L1 loss between predicted and actual errors

#### GAN Losses (Optional):
1. **LSGAN**: Least squares GAN with margin
2. **Hinge GAN**: Hinge loss formulation
3. **WGAN-GP**: Wasserstein GAN with gradient penalty

### Training Loop
```python
def training_step(self, batch, batch_idx):
    # Forward pass
    results = self.forward(batch)
    
    # Calculate reconstruction loss
    rec_loss_dict = self.compute_loss(results, batch)
    
    # Manual optimization
    g_opt = self.optimizers()
    g_opt.zero_grad()
    self.manual_backward(rec_loss_dict['loss'])
    g_opt.step()
```

### Optimization
- **Optimizer**: AdamW with configurable learning rate and weight decay
- **Learning Rate Scheduling**: Implicit through epoch-based training

## Evaluation Metrics

### 1. MPJPE (Mean Per Joint Position Error)
Average Euclidean distance between predicted and ground truth joint positions:
```
MPJPE = mean(||pred_joints - gt_joints||₂)
```

### 2. 3DPCK@40mm
Percentage of joints within 40mm of ground truth:
```
3DPCK@40mm = mean(||pred_joints - gt_joints||₂ ≤ 40mm)
```

### 3. Loss Components
- Joint position loss (L1)
- Error prediction loss

## Training Execution

### Command Line Interface
Training is executed through shell scripts that call `run_model.py`:

```bash
torchrun --nproc_per_node=2 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand/omnihand_rtm_collected_temporal.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "experiment_name" \
    --precision 32
```

### Distributed Training
- **Multi-GPU**: DDP (Distributed Data Parallel) strategy
- **Precision**: Full (32-bit), Mixed (16-bit), or BFloat16
- **Checkpointing**: Automatic model saving with metric monitoring

## Key Innovations

### 1. Doppler Velocity Accumulation
Models the physical principle that Doppler (velocity) information from earlier frames influences spatial feature extraction in later frames, specifically designed for mmWave radar data characteristics.

### 2. Cross-Stage Partial Architecture
Efficient 3D CNN design that balances computational cost with feature extraction capability.

### 3. Multi-Modal Temporal Processing
Three distinct approaches to temporal sequence modeling, each optimized for different aspects of mmWave data processing.

### 4. Error-Aware Regression
Joint prediction of 3D coordinates and per-joint error estimates for uncertainty quantification.

## Usage Examples

### Training a Base Model
```bash
./scripts/omnihand/pretrain_omnihand_rtm.sh
```

### Training with Temporal Processing
```bash
./scripts/omnihand/pretrain_omnihand_rtm_temporal.sh
```

### Configuration Example
```yaml
model_cfg:
  target: src.model.omnihand.OmniHand
  params:
    cfg:
      use_simulator: false
      learnable_weights: true
      backbone: 
        target: src.model.encoder.cubenet_rtm_temporal.CSPRTMTemporalEncoder3D
        params: 
          in_channels: 64
          base_channels: 64
          stage_channels: [128, 256, 512, 1024]
          temporal_frames: 5
          num_velocity_heads: 8
```