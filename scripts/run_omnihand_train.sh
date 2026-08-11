#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment_config="${MMPRISM_EXPERIMENT_CONFIG:-$project_root/configs/examples/omnihand_train_smoke_experiment.yaml}"
task_config="${MMPRISM_OMNIHAND_CONFIG:-$project_root/configs/examples/omnihand_train_smoke.yaml}"

: "${MMPRISM_TRAIN_MANIFEST:?Set MMPRISM_TRAIN_MANIFEST to a model-ready training manifest}"
: "${MMPRISM_VALIDATION_MANIFEST:?Set MMPRISM_VALIDATION_MANIFEST to a model-ready validation manifest}"

cd "$project_root"
exec uv run --frozen --extra train mmprism omnihand-train \
  "$experiment_config" \
  "$task_config" \
  --train-manifest "$MMPRISM_TRAIN_MANIFEST" \
  --validation-manifest "$MMPRISM_VALIDATION_MANIFEST" \
  --project-root "$project_root"
