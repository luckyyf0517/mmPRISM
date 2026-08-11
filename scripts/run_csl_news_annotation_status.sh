#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="${MMPRISM_CSL_NEWS_ANNOTATION_CONFIG:-$project_root/configs/data/csl_news_rtmw3d_overnight.yaml}"
integrity_registry="${MMPRISM_CSL_NEWS_INTEGRITY_REGISTRY:-/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2/registry.json}"
report_root="${MMPRISM_CSL_NEWS_STATUS_ROOT:-/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/reports}"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$report_root/status_${timestamp}.json"

mkdir -p "$report_root"
cd "$project_root"
uv run --frozen mmprism csl-news-annotation-status \
  "$config" \
  --project-root "$project_root" \
  --sample-validate 3 \
  --recent-window 200 \
  --integrity-registry "$integrity_registry" \
  --output "$output" \
  >/dev/null

echo "Wrote CSL-News annotation status: $output"
