#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="${MMPRISM_CSL_NEWS_INTEGRITY_CONFIG:-$project_root/configs/data/csl_news_source_integrity.yaml}"
export MMPRISM_DATA_ROOT="${MMPRISM_DATA_ROOT:-/mnt/gfs/yanyifan/mmPRISM}"

cd "$project_root"
exec uv run --frozen mmprism csl-news-integrity-scan \
  "$config" \
  --project-root "$project_root" \
  "$@"
