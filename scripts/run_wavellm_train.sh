#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment_config="${MMPRISM_EXPERIMENT_CONFIG:-$project_root/configs/examples/wavellm_train_smoke_experiment.yaml}"
task_config="${MMPRISM_WAVELLM_CONFIG:-$project_root/configs/examples/wavellm_train_smoke.yaml}"
asset_config="${MMPRISM_MT5_ASSET_CONFIG:-$project_root/configs/models/mt5_base_v1.yaml}"
model_root="${MMPRISM_MT5_MODEL_ROOT:-$project_root/pretrained_models/mt5_base_v1}"

: "${MMPRISM_TRAIN_MANIFEST:?Set MMPRISM_TRAIN_MANIFEST to a model-ready training manifest}"
: "${MMPRISM_VALIDATION_MANIFEST:?Set MMPRISM_VALIDATION_MANIFEST to a model-ready validation manifest}"

cd "$project_root"
exec uv run --frozen --extra train mmprism wavellm-train \
  "$experiment_config" \
  "$task_config" \
  --model-assets "$asset_config" \
  --model-root "$model_root" \
  --train-manifest "$MMPRISM_TRAIN_MANIFEST" \
  --validation-manifest "$MMPRISM_VALIDATION_MANIFEST" \
  --project-root "$project_root"
