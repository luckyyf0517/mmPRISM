#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${MMPRISM_MODEL_ROOT:-${project_root}/pretrained_models}"
# Direct HTTP was reliable for both pinned repositories on the reference host.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

cd "${project_root}"
uv run --frozen --extra evaluation mmprism models-download \
  configs/models/evaluation_models_v1.yaml \
  --output-root "${output_root}"
