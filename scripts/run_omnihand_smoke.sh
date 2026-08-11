#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact_root="${MMPRISM_ARTIFACT_ROOT:-${project_root}/artifacts}"
device="${MMPRISM_DEVICE:-auto}"

cd "${project_root}"
uv run --frozen --extra train mmprism omnihand-smoke \
  configs/examples/omnihand_smoke.yaml \
  --device "${device}" \
  --output "${artifact_root}/omnihand_cubenet_v1.json"
