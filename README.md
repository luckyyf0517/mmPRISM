# OmniHand: Millimeter Wave to Text Generation

This repository contains a PyTorch Lightning implementation for fine-tuning large language models (LLMs) with Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA, for millimeter wave signal to text generation.

## Overview

The implementation allows you to:

1. Fine-tune pre-trained language models (supporting Phi-3 and MT5) using LoRA
2. Process millimeter wave time series data using a pre-trained encoder
3. Generate text descriptions from millimeter wave signals
4. Extract and process pose features from millimeter wave data

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Transformers 4.30+
- PEFT 0.5+
- Wandb (for logging)
- DeepSpeed (for distributed training)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config/               # Configuration files for different models and training settings
├── dataset/             # Dataset files and annotations
├── scripts/             # Training and utility scripts
│   ├── train.sh        # Main training script for Phi-3 model
│   ├── train_mt5.sh    # Training script for MT5 model
│   ├── debug.sh        # Debug script for development
│   └── pretrain.sh     # Pretraining script
├── src/
│   ├── data/           # Dataset processing code
│   ├── model/          # Model implementations
│   │   ├── encoder/    # Millimeter wave signal encoder
│   │   └── llm/        # Language model wrappers
│   └── trainer/        # Training implementations
├── run_peft.py         # Main PEFT fine-tuning script
├── run_model.py        # Model inference script
├── run_extract_feature.py  # Feature extraction script
└── run_simulation.py   # Simulation and testing script
```

## Main Scripts

### Training Scripts

- `scripts/train.sh`: Main training script for Phi-3 model
  ```bash
  # Example usage with DeepSpeed on 2 GPUs
  deepspeed --include localhost:0,1 run_peft.py \
      --config config/wavellm_phi3.yaml \
      --batch-size 8 \
      --max-epochs 5
  ```

- `scripts/train_mt5.sh`: Training script for MT5 model
  ```bash
  # Example usage with DeepSpeed on 2 GPUs
  deepspeed --include localhost:0,1 run_peft.py \
      --config config/wavellm_mt5.yaml \
      --batch-size 24 \
      --max-epochs 10
  ```

### Data Processing

- `run_extract_feature.py`: Extract features from millimeter wave signals
- `run_csl_news_annotation.py`: Process and annotate CSL News dataset
- `info.py`: Dataset statistics and information

### Inference and Visualization

- `run_inference.py`: Model inference script
- `attention_visualize.py`: Visualize attention patterns
- `make_video.py`: Generate visualization videos
- `view_evaluation.ipynb`: Jupyter notebook for evaluation analysis

## Dataset

The implementation includes two main dataset classes:

1. `mmSingleImageDataset`: For single frame processing
   - Supports pose and feature extraction modes
   - Processes arm and hand joints
   - Handles 3D point data and velocities

2. `CslNewsDataset`: For sequence data processing
   - Supports pose, feature, and pose prediction modes
   - Handles time series data with captions
   - Includes data augmentation and preprocessing

## Training Configuration

The configuration files in `config/` directory contain all the parameters for training:

- Model configuration (model type, path, etc.)
- Signal encoder configuration
- Data configuration
- Training configuration
- PEFT configuration (LoRA parameters)
- DeepSpeed configuration

## Advanced Features

### Distributed Training

The project supports distributed training using DeepSpeed with the following features:
- ZeRO optimization (configurable stages)
- Mixed precision training (bf16/fp16)
- Gradient accumulation
- Multi-GPU support

### Model Support

Currently supported models:
- Phi-3 (default)
- MT5
- Extensible to other HuggingFace models

## License

This project is licensed under the MIT License - see the LICENSE file for details. 