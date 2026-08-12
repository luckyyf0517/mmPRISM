#!/usr/bin/env bash

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="${MMPRISM_CSL_NEWS_ANNOTATION_CONFIG:-$project_root/configs/data/csl_news_rtmw3d_overnight.yaml}"
registry="${MMPRISM_CSL_NEWS_INTEGRITY_REGISTRY:-/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2/registry.json}"

if (($# < 1)); then
  echo "Usage: $0 init|pause|resume|status [scheduler options]" >&2
  exit 2
fi

action="$1"
shift
case "$action" in
  init) command="csl-news-scheduler-init" ;;
  pause) command="csl-news-scheduler-pause" ;;
  resume) command="csl-news-scheduler-resume" ;;
  status) command="csl-news-scheduler-status" ;;
  *)
    echo "Unsupported scheduler action: $action" >&2
    exit 2
    ;;
esac

arguments=("$config" --project-root "$project_root")
if [[ "$action" == "status" ]]; then
  arguments+=(--integrity-registry "$registry")
fi

cd "$project_root"
exec uv run --frozen mmprism "$command" "${arguments[@]}" "$@"
