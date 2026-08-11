#!/usr/bin/env bash

set -euo pipefail

profile="${1:-research}"

case "${profile}" in
  foundation)
    sync_args=()
    ;;
  research)
    sync_args=(
      --extra train
      --extra radar
      --extra evaluation
      --extra tracking
      --extra visualization
      --extra annotation
    )
    ;;
  distributed)
    sync_args=(
      --extra train
      --extra radar
      --extra evaluation
      --extra tracking
      --extra visualization
      --extra annotation
      --extra distributed
    )
    ;;
  *)
    echo "Usage: scripts/bootstrap_env.sh [foundation|research|distributed]" >&2
    exit 2
    ;;
esac

uv sync --frozen "${sync_args[@]}"
uv run mmprism doctor
