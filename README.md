# PEFT Fine-tuning for Millimeter Wave to Text Generation

This repository contains a PyTorch Lightning implementation for fine-tuning large language models (LLMs) with Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA, for millimeter wave signal to text generation.

## Overview

The implementation allows you to:

1. Fine-tune a pre-trained language model (currently supporting Phi-3) using LoRA
2. Process millimeter wave time series data using a pre-trained encoder
3. Generate text descriptions from millimeter wave signals

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- Transformers 4.30+
- PEFT 0.5+
- Wandb (for logging)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

- `src/trainer/wavellm.py`: PyTorch Lightning module for PEFT fine-tuning
- `src/trainer/train_peft.py`: Training script
- `src/data/dataset.py`: Dataset classes for millimeter wave data
- `src/model/encoder/encoder.py`: Encoder for millimeter wave signals
- `src/model/llm/phi3_model.py`: Phi-3 model wrapper
- `configs/peft_finetune.yaml`: Configuration file for fine-tuning

## Usage

### Training

To train the model, use the following command:

```bash
python src/trainer/train_peft.py --config configs/peft_finetune.yaml --output_dir outputs/peft_finetune
```

### Configuration

The configuration file (`configs/peft_finetune.yaml`) contains all the parameters for training:

- Model configuration (model type, path, etc.)
- Signal encoder configuration
- Data configuration
- Training configuration
- PEFT configuration (LoRA parameters)

### Dataset

The implementation includes a `mmWaveSequenceDataset` class that handles time series millimeter wave data. The dataset expects:

1. A JSON file with data paths
2. A JSON file with captions (optional)

The dataset processes pose data to extract arm and hand joints, and formats it for the model.

## How It Works

1. **Data Processing**: The millimeter wave time series data is processed by the encoder to extract features.
2. **LoRA Fine-tuning**: The language model is fine-tuned using LoRA, which adds small trainable rank decomposition matrices to existing weights.
3. **Text Generation**: The model generates text descriptions from the processed millimeter wave signals.

## Customization

### Adding a New Model

To add a new model, create a new model wrapper class in `src/model/llm/` and register it in `src/model/llm/model_factory.py`.

### Modifying the Encoder

The encoder in `src/model/encoder/encoder.py` can be modified to handle different types of millimeter wave data.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 