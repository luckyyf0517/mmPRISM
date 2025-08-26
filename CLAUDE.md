# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two main systems:
1. **WaveLLM**: PyTorch Lightning implementation for fine-tuning large language models (LLMs) with Parameter-Efficient Fine-Tuning (PEFT) techniques for millimeter wave signal to text generation
2. **OmniHand**: Deep learning system for 3D hand pose estimation from millimeter wave radar data

The OmniHand system has been recently enhanced with temporal processing capabilities including a TVAN-inspired (Temporal Velocity Accumulation Network) architecture for improved accuracy in processing sequences of mmwave data.

## Code Architecture and Structure

```
.
├── config/               # Configuration files for different models and training settings
│   ├── omnihand_tvan.yaml                    # Configuration for TVAN-based temporal processing
│   └── omnihand_rtm_collected_temporal.yaml  # Configuration for temporal RTM processing
├── dataset/             # Dataset files and annotations
├── scripts/             # Training and utility scripts
│   ├── train.sh                           # Main training script for Phi-3 model
│   ├── train_mt5.sh                       # Training script for MT5 model
│   ├── debug.sh                           # Debug script for development
│   ├── pretrain.sh                        # Pretraining script
│   ├── pretrain_omnihand_tvan.sh          # Training script for TVAN-based model
│   └── pretrain_omnihand_rtm_temporal.sh  # Training script for temporal RTM model
├── src/
│   ├── data/           # Dataset processing code
│   ├── model/          # Model implementations
│   │   ├── encoder/    # Millimeter wave signal encoder
│   │   │   ├── cubenet_rtm_temporal.py    # Temporal extension with TVAN-inspired processing
│   │   └── llm/        # Language model wrappers
│   └── trainer/        # Training implementations
├── run_peft.py         # Main PEFT fine-tuning script
├── run_model.py        # Model inference script
├── run_extract_feature.py  # Feature extraction script
└── run_simulation.py   # Simulation and testing script
```

## Common Development Commands

### WaveLLM Training Commands

1. **Training MT5 model on daily dataset**:
   ```bash
   deepspeed --include localhost:0,1 \
       --master_port 12345 \
       run_peft.py \
       --config config/wavellm/wavellm_mt5_daily_features.yaml \
       --batch-size 64 \
       --max-epochs 10 \
       --gradient-accumulation-steps 8 \
       --version "wavellm_mt5_daily_features_0813" \
       --dtype bf16 \
       --seed 42 \
       --zero_stage 2 \
       --resume-checkpoint "log/archived/wavellm_mt5_gt_pose_0523/last.ckpt" \
       --reset
   ```

2. **Evaluating MT5 model**:
   ```bash
   deepspeed --include localhost:0,1 \
       --master_port 12346 \
       run_peft.py \
       --config config/wavellm/wavellm_mt5_daily_features.yaml \
       --batch-size 24 \
       --version "wavellm_mt5_daily_features_0813_eval" \
       --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_features_0813/last.ckpt"  \
       --dtype bf16 \
       --zero_stage 2 \
       --test
   ```

3. **Running evaluation script**:
   ```bash
   python run_evaluation.py --results_dir path/to/results --output_dir path/to/output
   ```

### OmniHand Training Commands

1. **Training with TVAN-inspired temporal processing**:
   ```bash
   ./scripts/pretrain_omnihand_tvan.sh
   ```

2. **Training with temporal RTM processing**:
   ```bash
   ./scripts/pretrain_omnihand_rtm_temporal.sh
   ```

### Key Python Scripts

**WaveLLM Scripts:**
- `run_peft.py`: Main entry point for training and evaluation
- `run_evaluation.py`: Calculates metrics from test results
- `src/model/trainer.py`: Core PyTorch Lightning training module
- `src/model/llm/model_factory.py`: Factory for creating language models
- `src/model/encoder/pose_encoder.py`: Pose encoder using GCN networks
- `src/data/dataset.py`: Dataset implementations for different data formats

**OmniHand Scripts:**
- `src/model/omnihand.py`: Main OmniHand model implementation
- `src/model/encoder/cubenet_rtm.py`: Base CubeNet RTM encoder
- `src/model/encoder/cubenet_rtm_temporal.py`: Temporal extension with TVAN-inspired processing for mmwave sequences
- `src/fmcw/simulator.py`: Radar signal simulator and processor

## Architecture Overview

### WaveLLM System

#### Core Components

1. **WaveLLMTrainer (src/model/trainer.py)**: 
   - PyTorch Lightning module for PEFT fine-tuning with LoRA
   - Integrates millimeter wave signal encoders with language models
   - Supports multiple modalities (features, pose data)
   - Handles training, validation, and testing workflows

2. **Model Factory (src/model/llm/model_factory.py)**:
   - Factory pattern for creating different LLM types (Phi-3, MT5)
   - Handles model loading and configuration

3. **Pose Encoder (src/model/encoder/pose_encoder.py)**:
   - Processes pose data using Spatio-Temporal GCN networks
   - Supports body and hand pose processing
   - Projects pose features to LLM hidden dimensions

4. **Dataset Classes (src/data/dataset.py)**:
   - Multiple dataset implementations for different data formats
   - Supports sequence and single-frame data processing
   - Handles data normalization, augmentation, and preprocessing

#### Data Flow

1. **Training**: 
   - Data is loaded through dataset classes
   - Pose/features are processed by encoders
   - Encoded features are fed to LLM through WaveLLMTrainer
   - Loss is computed and backpropagated

2. **Inference**:
   - Model loads checkpoint
   - Input data is processed through encoders
   - LLM generates text based on encoded features

#### Key Features

1. **Distributed Training**: 
   - Uses DeepSpeed with ZeRO optimization
   - Supports mixed precision training (bf16/fp16)
   - Multi-GPU support

2. **Modality Support**:
   - Pose data processing with GCN networks
   - Pre-computed feature processing
   - Multiple modality fusion strategies

3. **Model Support**:
   - Phi-3 and MT5 language models
   - PEFT/LoRA fine-tuning
   - Extensible to other HuggingFace models

### OmniHand System

#### Core Components

1. **OmniHand Model (src/model/omnihand.py)**:
   - PyTorch Lightning module for 3D hand pose estimation
   - Supports both simulated and real mmwave data
   - Integrates with various backbone encoders
   - Optional GAN-based discriminator for improved realism

2. **Encoders**:
   - **MMHand Encoder (src/model/encoder/mmhand_encoder.py)**: Simplified architecture with attention mechanisms
   - **CubeNet Encoder (src/model/encoder/cubenet.py)**: Traditional 3D CNN with ResNet-style blocks
   - **CubeNet RTM Encoder (src/model/encoder/cubenet_rtm.py)**: Advanced CSP/PAFPN architecture
   - **CubeNet RTM Temporal Encoder (src/model/encoder/cubenet_rtm_temporal.py)**: Temporal extension for sequence processing
   - **Temporal Velocity Accumulation Network (TVAN) (src/model/encoder/tvan.py)**: Novel module that captures intrinsic temporal relationships where earlier frames' Doppler (velocity) information accumulates and enhances later frames' spatial processing

3. **Radar Simulator and Processor (src/fmcw/simulator.py)**:
   - Simulates mmwave radar signals from 3D point clouds
   - Processes raw radar data through range, Doppler, and beamforming
   - Outputs 5D mmwave cubes: [B, 64, 32, 32, 32] (Batch, Doppler, Range, Width, Height)

4. **Dataset Classes (src/data/dataset.py)**:
   - CollectedDailyDataset with temporal processing support
   - Configurable temporal frame sequences
   - Support for both single-frame and multi-frame processing

5. **Enhanced Temporal Encoder (src/model/encoder/cubenet_rtm_temporal.py)**:
   - CSPRTMTemporalEncoder3D with TVAN-inspired temporal processing
   - Processes sequences of mmwave cubes: [B, T, 64, 32, 32, 32]
   - Uses shared spatial encoder for all frames with temporal attention
   - Incorporates Temporal Velocity Accumulation Network concepts for enhanced feature extraction
   - Outputs single global feature vector: [B, 256] for hand pose estimation

#### Temporal Processing Features

1. **Dataset Modifications**:
   - Added `use_temporal` flag for enabling temporal processing
   - Configurable `num_temporal_frames` (default: 5)
   - Loads sequences of mmwave frames: [T, num_chirps, num_antenna, num_samples]

2. **Temporal Encoder (CSPRTMTemporalEncoder3D)**:
   - Processes sequences of mmwave cubes: [B, T, 64, 32, 32, 32]
   - Uses shared spatial encoder for all frames
   - Supports temporal attention or convolutional fusion
   - Outputs single global feature vector: [B, 256] (updated for TVAN-inspired processing)

3. **Temporal Velocity Accumulation Network (TVAN)**:
   - **Novel Contribution**: Captures intrinsic temporal relationships in mmwave data
   - Earlier frames' Doppler (velocity) information accumulates to provide contextual enhancement
   - Models natural temporal coherence where velocity context guides spatial feature extraction
   - Key Innovation: Velocity accumulation LSTM processes Doppler information through time
   - Velocity-guided spatial processor uses accumulated context to enhance spatial features
   - Temporal attention fusion weights different frame contributions
   - Output: [B, 256] enhanced temporal features for hand pose estimation

4. **Enhanced Temporal Encoder (CSPRTMTemporalEncoder3D)**:
   - **TVAN-Inspired Innovation**: Incorporates key concepts from Temporal Velocity Accumulation Network
   - Shared spatial encoder processes all frames in the temporal sequence
   - LSTM-based velocity accumulator captures temporal relationships across frames
   - Accumulated temporal context directly enhances spatial feature processing
   - Outputs enhanced global feature vector: [B, 256] for improved hand pose estimation

5. **Model Integration**:
   - OmniHand automatically detects temporal input tensors
   - Routes temporal data through appropriate backbone
   - Maintains backward compatibility with single-frame models

#### Data Flow

1. **Single-Frame Processing**:
   - Raw mmwave data: [B, num_chirps, num_antenna, num_samples]
   - Processor output: [B, 64, 32, 32, 32]
   - Encoder features: [B, 1024]
   - Hand pose output: [B, 2, 24, 3]

2. **Temporal Processing**:
   - Raw mmwave sequences: [B, T, num_chirps, num_antenna, num_samples]
   - Per-frame processing through Processor
   - Temporal encoder output: [B, 256] (enhanced features from TVAN-inspired processing)
   - Hand pose output: [B, 2, 24, 3] (single pose from temporal sequence)

#### Key Features

1. **Modular Encoder Architecture**:
   - Pluggable backbone encoders
   - Configurable through YAML files
   - Support for attention mechanisms

2. **Temporal Processing**:
   - Sequence-based mmwave processing
   - Configurable number of temporal frames
   - Attention-based temporal fusion

3. **Flexible Data Handling**:
   - Support for simulated and real mmwave data
   - Configurable data augmentation
   - Multiple dataset implementations

## Evaluation and Metrics

### WaveLLM Evaluation
The evaluation pipeline includes:
- Traditional metrics (BLEU, ROUGE-L)
- Semantic similarity metrics (SBERT, SimCSE)
- Custom evaluation script in `run_evaluation.py`

### OmniHand Evaluation
- 3D Pose estimation metrics (MPJPE, 3DPCK@40mm)
- Reconstruction loss monitoring
- GAN discriminator accuracy (when enabled)