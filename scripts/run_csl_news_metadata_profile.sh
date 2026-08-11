#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
metadata_root="${MMPRISM_CSL_NEWS_METADATA_ROOT:-/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121/metadata}"
report_root="${MMPRISM_CSL_NEWS_METADATA_REPORT_ROOT:-/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_metadata_profile_v1}"
source_revision="${MMPRISM_CSL_NEWS_REVISION:-3a0601210333fe760efd09b5d9e2ae5f341ce339}"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output="$report_root/profile_${timestamp}.json"

mkdir -p "$report_root"
cd "$project_root"
uv run --frozen mmprism csl-news-metadata-profile \
  --labels-json "$metadata_root/CSL_News_Labels.json" \
  --labels-csv "$metadata_root/CSL_News_Labels.csv" \
  --dataset-card "$metadata_root/README.md" \
  --source-id "huggingface:ZechengLi19/CSL-News" \
  --source-revision "$source_revision" \
  --output "$output" \
  >/dev/null

echo "Wrote CSL-News metadata profile: $output"
