#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="${MMPRISM_CSL_NEWS_SOURCE_MANIFEST_CONFIG:-$project_root/configs/data/csl_news_source_manifest_available.yaml}"

: "${MMPRISM_DATA_ROOT:?Set MMPRISM_DATA_ROOT to the canonical mmPRISM data root}"

cd "$project_root"
uv run --frozen mmprism csl-news-source-manifest \
  "$config" \
  --project-root "$project_root"
