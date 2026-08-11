#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${MMPRISM_MT5_MODEL_ROOT:-${project_root}/pretrained_models/mt5_base_v1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "${project_root}"
uv run --frozen --extra train mmprism models-download \
  configs/models/mt5_base_v1.yaml \
  --output-root "${output_root}"
