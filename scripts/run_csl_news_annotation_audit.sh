#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="${MMPRISM_CSL_NEWS_ANNOTATION_CONFIG:-$project_root/configs/data/csl_news_rtmw3d_overnight.yaml}"
audit_root="${MMPRISM_CSL_NEWS_ANNOTATION_AUDIT_ROOT:-/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/identity_audits}"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$audit_root/audit_${timestamp}.json"

mkdir -p "$audit_root"
cd "$project_root"
set +e
uv run --frozen mmprism csl-news-annotation-audit \
  "$config" \
  --project-root "$project_root" \
  --output "$output" \
  >/dev/null
status=$?
set -e

echo "Wrote CSL-News annotation identity audit: $output"
exit "$status"
