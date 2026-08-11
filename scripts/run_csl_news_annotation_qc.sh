#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="${MMPRISM_CSL_NEWS_ANNOTATION_CONFIG:-$project_root/configs/data/csl_news_rtmw3d_overnight.yaml}"
qc_root="${MMPRISM_CSL_NEWS_QC_ROOT:-/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/qc}"
sample_count="${MMPRISM_CSL_NEWS_QC_SAMPLES:-100}"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$qc_root/qc_${timestamp}.json"

mkdir -p "$qc_root"
cd "$project_root"
uv run --frozen mmprism csl-news-annotation-qc \
  "$config" \
  --project-root "$project_root" \
  --sample-count "$sample_count" \
  --output "$output" \
  >/dev/null

echo "Wrote CSL-News annotation QC: $output"
