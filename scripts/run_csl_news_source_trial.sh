#!/usr/bin/env bash
set -euo pipefail

artifact_root="${MMPRISM_CSL_NEWS_TRIAL_ROOT:-/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/source_trial_v1}"
integrity_config="${MMPRISM_CSL_NEWS_INTEGRITY_CONFIG:-configs/data/csl_news_source_integrity.yaml}"
decode_samples="${MMPRISM_CSL_NEWS_DECODE_SAMPLES:-3}"
archive_id="${MMPRISM_CSL_NEWS_TRIAL_ARCHIVE_ID:-}"
export MMPRISM_DATA_ROOT="${MMPRISM_DATA_ROOT:-/mnt/gfs/yanyifan/mmPRISM}"

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

selection_file=$(mktemp)
trap 'rm -f "$selection_file"' EXIT
selection_args=(csl-news-integrity-select "$integrity_config")
if [[ -n "$archive_id" ]]; then
  selection_args+=(--archive-id "$archive_id")
fi
uv run --frozen mmprism "${selection_args[@]}" >"$selection_file"

read_json_field() {
  uv run --frozen python -c \
    'import json, sys; value=json.load(open(sys.argv[1], encoding="utf-8")); print(value[sys.argv[2]][sys.argv[3]])' \
    "$selection_file" "$1" "$2"
}

archive_path=$(read_json_field archive path)
archive_name=$(read_json_field archive archive_name)
labels_path=$(read_json_field source labels_path)
source_id=$(read_json_field source source_ref)
registry_sha256=$(read_json_field registry sha256)
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$artifact_root/${timestamp}_${archive_name%.zip}_${registry_sha256:0:12}"
mkdir -p "$run_dir"
mv "$selection_file" "$run_dir/source_selection.json"
trap - EXIT

printf '%s\n' \
  "archive=$archive_path" \
  "labels=$labels_path" \
  "source_id=$source_id" \
  "integrity_config=$integrity_config" \
  "integrity_registry_sha256=$registry_sha256" \
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
