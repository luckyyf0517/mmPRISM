#!/usr/bin/env bash
set -euo pipefail

source_root="${MMPRISM_CSL_NEWS_ROOT:-/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121}"
artifact_root="${MMPRISM_CSL_NEWS_TRIAL_ROOT:-/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1}"
source_id="ZechengLi19/CSL-News@3a0601210333fe760efd09b5d9e2ae5f341ce339"
decode_samples="${MMPRISM_CSL_NEWS_DECODE_SAMPLES:-3}"
archive_path="${MMPRISM_CSL_NEWS_TRIAL_ARCHIVE:-}"

if [[ -z "$archive_path" ]]; then
  archive_path=$(find "$source_root/rgb_archives" -maxdepth 1 -type f -name 'archive_*.zip' | sort | head -n 1)
fi

if [[ -z "$archive_path" || ! -f "$archive_path" ]]; then
  echo "No complete CSL-News archive is available under $source_root/rgb_archives" >&2
  exit 2
fi

labels_path="$source_root/metadata/CSL_News_Labels.json"
if [[ ! -f "$labels_path" ]]; then
  echo "Complete CSL-News labels are not available: $labels_path" >&2
  exit 2
fi

archive_name=$(basename "$archive_path" .zip)
run_dir="$artifact_root/20260812_${archive_name}"
mkdir -p "$run_dir"

printf '%s\n' \
  "archive=$archive_path" \
  "labels=$labels_path" \
  "source_id=$source_id" \
  "decode_samples=$decode_samples" \
  >"$run_dir/resolved_inputs.txt"

git rev-parse HEAD >"$run_dir/git_commit.txt"
git status --porcelain >"$run_dir/git_status.txt"
sha256sum uv.lock >"$run_dir/uv_lock.sha256"
uv run --frozen mmprism doctor >"$run_dir/runtime.json"

printf '%q ' uv run --frozen mmprism csl-news-audit "$archive_path" \
  --labels "$labels_path" \
  --source-id "$source_id" \
  --output "$run_dir/source_audit.json" \
  --decode-samples "$decode_samples" \
  --scratch-dir "$run_dir"
printf '\n'

uv run --frozen mmprism csl-news-audit "$archive_path" \
  --labels "$labels_path" \
  --source-id "$source_id" \
  --output "$run_dir/source_audit.json" \
  --decode-samples "$decode_samples" \
  --scratch-dir "$run_dir" \
  >"$run_dir/source_audit.stdout.json"

printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$run_dir/SUCCESS"
echo "CSL-News source trial passed: $run_dir/source_audit.json"
