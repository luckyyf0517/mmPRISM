#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
experiment_config="${MMPRISM_EXPERIMENT_CONFIG:-$project_root/configs/examples/omnihand_train_smoke_experiment.yaml}"
task_config="${MMPRISM_OMNIHAND_CONFIG:-$project_root/configs/examples/omnihand_train_smoke.yaml}"

: "${MMPRISM_TRAIN_MANIFEST:?Set MMPRISM_TRAIN_MANIFEST to a model-ready training manifest}"
: "${MMPRISM_VALIDATION_MANIFEST:?Set MMPRISM_VALIDATION_MANIFEST to a model-ready validation manifest}"
: "${MMPRISM_SPLIT_ASSIGNMENTS:?Set MMPRISM_SPLIT_ASSIGNMENTS to canonical split assignments}"

resume_args=()
if [[ -n "${MMPRISM_RESUME_STATE_METADATA:-}" || -n "${MMPRISM_RESUME_STATE_TENSORS:-}" ]]; then
  : "${MMPRISM_RESUME_STATE_METADATA:?Set both MMPRISM_RESUME_STATE_METADATA and MMPRISM_RESUME_STATE_TENSORS}"
  : "${MMPRISM_RESUME_STATE_TENSORS:?Set both MMPRISM_RESUME_STATE_METADATA and MMPRISM_RESUME_STATE_TENSORS}"
  resume_args=(
    --resume-state-metadata "$MMPRISM_RESUME_STATE_METADATA"
    --resume-state-tensors "$MMPRISM_RESUME_STATE_TENSORS"
  )
fi

cd "$project_root"
exec uv run --frozen --extra train mmprism omnihand-train \
  "$experiment_config" \
  "$task_config" \
  --train-manifest "$MMPRISM_TRAIN_MANIFEST" \
  --validation-manifest "$MMPRISM_VALIDATION_MANIFEST" \
  --split-assignments "$MMPRISM_SPLIT_ASSIGNMENTS" \
  "${resume_args[@]}" \
  --project-root "$project_root"
