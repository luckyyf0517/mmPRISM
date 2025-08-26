# WaveLLM Architecture Documentation

## Overview

WaveLLM is a multimodal language model framework designed for processing millimeter wave (mmWave) radar data combined with natural language processing tasks. The system integrates mmWave signal processing, pose estimation, and large language models (LLMs) to enable applications such as sign language translation.

## System Architecture

### Core Components

1. **Data Processing Pipeline**
   - mmWave radar signal processing and feature extraction
   - 3D pose estimation from radar data
   - Multimodal data fusion for LLM input

2. **Neural Network Architecture**
   - Pose encoder using Spatio-Temporal GCN networks
   - Feature projection networks
   - Large language models (MT5, Phi-3)
   - PEFT/LoRA fine-tuning capabilities

3. **Training Framework**
   - PyTorch Lightning for distributed training
   - DeepSpeed optimization for large model training
   - Parameter-Efficient Fine-Tuning (PEFT) support

## Data Flow

### 1. Input Data
The system accepts multiple types of input data:
- **Pose Data**: 3D joint coordinates [B, T, 2, 24, 3] (batch, time, hands, joints, coordinates)
- **Feature Data**: Pre-computed features [B, T, feature_dim]
- **Text Data**: Natural language captions/annotations

### 2. Pose Encoding
The `HandPoseEncoder` processes 3D pose data using Spatio-Temporal GCN networks:
1. **Graph Construction**: Creates body and hand graphs with spatial relationships
2. **Spatial GCN**: Processes spatial relationships between joints
3. **Temporal GCN**: Captures temporal dynamics across frames
4. **Feature Fusion**: Combines body and hand features with learned importance weights

### 3. Feature Processing
For pre-computed features:
1. **Convolutional Projection**: 1D convolutions to process temporal features
2. **Feature Gating**: Optional gating mechanism when combining with pose features

### 4. Multimodal Fusion
Features from different modalities are combined:
- Simple concatenation when only one modality is used
- Gated fusion when both pose and feature modalities are enabled

### 5. Language Model Integration
The fused multimodal features are injected into the LLM:
- **Wave Embeddings**: Multimodal features aligned with LLM hidden dimensions
- **Prompt Processing**: Text prompts tokenized and processed
- **Language Generation**: LLM generates text based on multimodal input

## Model Architecture

### WaveLLMTrainer Class
The main model class `WaveLLMTrainer` inherits from PyTorch Lightning's `LightningModule` and includes:

#### Key Components:
1. **Language Model**: MT5 or Phi-3 language models
2. **Pose Encoder**: Spatio-Temporal GCN networks for pose processing
3. **Feature Projector**: Convolutional networks for feature processing
4. **Tokenizer**: Tokenizer for language model input/output

#### Forward Pass:
```python
def forward(self, batch):
    # Get pose embeddings
    wave_embeds = self._get_wave_embeds(batch)
    
    # Prepare prompt texts
    prompts = [f"Translate hand sign language videos to Chinese:" for _ in range(len(batch['caption']))]
    
    # Tokenize prompt texts
    prompt_tokens = self.tokenizer(prompts, padding="longest", truncation=True, return_tensors="pt")
    
    # Tokenize target texts
    target_tokens = self.tokenizer(batch['caption'], padding="longest", truncation=True, return_tensors="pt")
    
    # Forward pass
    outputs = self.model(
        wave_embeds=wave_embeds,
        input_ids=prompt_tokens['input_ids'],
        attention_mask=prompt_tokens['attention_mask'],
        labels=target_tokens['input_ids']
    )
```

### Pose Encoder (HandPoseEncoder)

#### Architecture:
1. **Input Projection**: Linear projection of 3D coordinates to hidden dimension
2. **Graph Construction**: Body (7 joints) and hand (21 joints) graphs
3. **Spatial GCN**: Processes spatial relationships with adaptive adjacency matrices
4. **Temporal GCN**: Captures temporal dynamics with temporal convolutions
5. **Feature Fusion**: Combines body and hand features with learned weights
6. **Final Projection**: Maps to LLM hidden dimension

#### Key Features:
- **Spatio-Temporal GCN**: Combines spatial and temporal processing in a single framework
- **Adaptive Graphs**: Learnable adjacency matrices for better representation
- **Part-based Processing**: Separate processing for body and hands with reference point connections
- **Feature Gating**: Learnable parameters for part importance weighting

### Language Models

#### Supported Models:
1. **MT5**: Multilingual T5 model for cross-lingual tasks
2. **Phi-3**: Lightweight but powerful language model from Microsoft

#### Model Customization:
- Custom model loading with configurable paths
- Hidden dimension alignment with multimodal encoders
- Support for different model sizes and configurations

## Training Framework

### Configuration System
Training is configured through YAML files that specify:
- Data paths and preprocessing options
- Model architecture parameters
- Training hyperparameters
- PEFT/LoRA configuration

### Data Interface
The `BaseDataInterface` handles:
- Loading sequence data with multiple modalities
- Configurable data augmentation (downsampling, upsampling)
- Modality selection (pose, features, raw pose, ground truth pose)
- Text annotation loading and preprocessing

### Training Process

#### Initialization:
1. **Model Creation**: Initialize LLM and multimodal encoders
2. **PEFT Setup**: Configure LoRA adapters if enabled
3. **Optimizer Configuration**: AdamW optimizer with cosine scheduling
4. **Distributed Training**: DeepSpeed with ZeRO optimization

#### Training Loop:
```python
def training_step(self, batch, batch_idx):
    outputs = self(batch)
    loss = outputs['loss']
    self.log('train/loss', loss, on_step=True, on_epoch=False, 
             prog_bar=True, sync_dist=True, batch_size=self.batch_size)
    return loss
```

#### Key Features:
- **DeepSpeed Integration**: ZeRO optimization for memory-efficient training
- **Mixed Precision**: BF16/FP16 training for performance
- **Gradient Accumulation**: Support for large batch training
- **Learning Rate Scheduling**: Cosine decay with warmup

### Parameter-Efficient Fine-Tuning (PEFT)

#### LoRA Configuration:
- **Rank (r)**: LoRA adapter rank (default: 16)
- **Alpha**: LoRA scaling factor (default: 32)
- **Target Modules**: Which layers to apply LoRA to (attention projections)
- **Dropout**: Regularization for LoRA adapters

#### Benefits:
- **Memory Efficiency**: Only train small adapter parameters
- **Fast Training**: Reduced computational requirements
- **Transfer Learning**: Easy adaptation to new tasks

## Configuration Details

### Data Configuration

#### Modalities:
```yaml
modalities:
  use_pred_pose: false      # Use predicted pose data
  use_features: true        # Use pre-computed features
  use_raw_pose: false       # Use raw pose with velocity
  use_gt_pose: false        # Use ground truth pose data
```

#### Pose Configuration:
```yaml
pose_config:
  pose_dir: "pred_poses_0602_rtm"  # Directory for pose data
  norm_pose: true                  # Normalize pose data
```

#### Feature Configuration:
```yaml
feature_config:
  feature_dir: "features_0602_rtm"  # Directory for feature data
  feature_dim: 1024                 # Feature dimension
```

### Model Configuration

#### Language Model:
```yaml
model_type: "mt5"                    # Model type (mt5 or phi3)
model:
  model_path: "huggingface/mt5-pretrained"  # Pretrained model path
  model_max_length: 2048                    # Maximum sequence length
```

#### PEFT Configuration:
```yaml
use_peft: false              # Enable PEFT
fix_llm: false               # Freeze LLM parameters
peft_config:
  r: 16                      # LoRA rank
  lora_alpha: 32             # LoRA alpha
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target layers
  lora_dropout: 0.05         # LoRA dropout
  bias: "none"               # Bias handling
```

#### Training Configuration:
```yaml
training:
  learning_rate: 8.0e-4      # Learning rate
  weight_decay: 1.0e-4       # Weight decay
  warmup_steps: 100          # Learning rate warmup steps
```

## Training Execution

### Command Line Interface
Training is executed through shell scripts that call `run_peft.py`:

```bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm/wavellm_mt5_daily_features.yaml \
    --batch-size 64 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "experiment_name" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset
```

### Distributed Training
- **Multi-GPU**: DeepSpeed with ZeRO optimization
- **Precision**: BF16/FP16 mixed precision training
- **Checkpointing**: Automatic model saving with versioning
- **Memory Optimization**: Gradient compression and partitioning

## Evaluation and Testing

### Test Mode
```bash
python run_peft.py \
    --config config/wavellm/wavellm_mt5_daily_features.yaml \
    --batch-size 24 \
    --version "experiment_name_eval" \
    --resume-checkpoint "path/to/checkpoint" \
    --dtype bf16 \
    --zero_stage 2 \
    --test
```

### Generation
- **Beam Search**: 4-beam decoding for better quality
- **Max Length**: Configurable maximum generation length
- **Sampling**: Deterministic generation with no sampling

## Key Innovations

### 1. Spatio-Temporal GCN for Pose Processing
- Novel application of GCN networks to mmWave pose data
- Adaptive graph learning for better representation
- Part-based processing with reference point connections

### 2. Multimodal Fusion Architecture
- Flexible modality selection and combination
- Gated fusion for optimal feature integration
- Alignment with LLM hidden dimensions

### 3. Efficient Training Framework
- DeepSpeed integration for large model training
- PEFT/LoRA for parameter-efficient adaptation
- Mixed precision training for performance

### 4. Cross-Lingual Capabilities
- Support for multilingual models (MT5)
- Configurable prompt engineering
- Flexible task adaptation

## Usage Examples

### Training with Features Only
```bash
./scripts/wavellm/train.sh
```

### Training with Pose Data
```bash
deepspeed --include localhost:0,1,2,3 \
    run_peft.py \
    --config config/wavellm/wavellm_mt5_daily.yaml \
    --batch-size 64 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_daily_0813" \
    --dtype bf16 \
    --seed 42 \
    --zero_stage 2 \
    --resume-checkpoint "log/archived/wavellm_mt5_gt_pose_0523/last.ckpt" \
    --reset
```

### Evaluation
```bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm/wavellm_mt5_daily_features.yaml \
    --batch-size 24 \
    --version "wavellm_mt5_daily_features_0813_eval" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_features_0813/last.ckpt"  \
    --dtype bf16 \
    --zero_stage 2 \
    --test
```