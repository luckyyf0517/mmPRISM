#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_root="${MMPRISM_MT5_MODEL_ROOT:-${project_root}/pretrained_models/mt5_base_v1}"
artifact_root="${MMPRISM_ARTIFACT_ROOT:-${project_root}/artifacts}"
device="${MMPRISM_DEVICE:-auto}"

cd "${project_root}"
uv run --frozen --extra train mmprism mt5-smoke \
  configs/examples/mt5_smoke.yaml \
  --model-assets configs/models/mt5_base_v1.yaml \
  --model-root "${model_root}" \
  --device "${device}" \
  --output "${artifact_root}/mt5_geometry_fusion_v1.json"
