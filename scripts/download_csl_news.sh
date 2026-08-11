#!/usr/bin/env bash
set -euo pipefail

REPO_ID="ZechengLi19/CSL-News"
REVISION="3a0601210333fe760efd09b5d9e2ae5f341ce339"
ARCHIVE_COUNT=436
FULL_DATASET_BYTES=935001573087
DEFAULT_RESERVE_BYTES=$((1024 * 1024 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage: scripts/download_csl_news.sh [options]

Options:
  --output-dir PATH   Incoming batch directory.
  --start N           First archive ID, inclusive (default: 1).
  --end N             Last archive ID, inclusive (default: 436).
  --workers N         Concurrent archive downloads (default: 16).
  --reserve-bytes N   Free-space reserve after compressed downloads (default: 1 TiB).
  --metadata-only     Download README and label files only.
  -h, --help          Show this help.

The downloader pins the official Hugging Face dataset revision, resumes partial
files, never extracts archives, and atomically renames completed downloads.
EOF
}

output_dir="${MMPRISM_CSL_NEWS_ROOT:-/mnt/gfs/yanyifan/mmPRISM/incoming/20260811_csl_news_hf_3a060121}"
start=1
end=$ARCHIVE_COUNT
workers=16
reserve_bytes=$DEFAULT_RESERVE_BYTES
metadata_only=0

while (($#)); do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --start)
      start="$2"
      shift 2
      ;;
    --end)
      end="$2"
      shift 2
      ;;
    --workers)
      workers="$2"
      shift 2
      ;;
    --reserve-bytes)
      reserve_bytes="$2"
      shift 2
      ;;
    --metadata-only)
      metadata_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((start < 1 || end > ARCHIVE_COUNT || start > end)); then
  echo "Archive range must satisfy 1 <= start <= end <= $ARCHIVE_COUNT" >&2
  exit 2
fi
if ((workers < 1)); then
  echo "Worker count must be positive" >&2
  exit 2
fi

metadata_dir="$output_dir/metadata"
archive_dir="$output_dir/rgb_archives"
log_dir="$output_dir/logs"
mkdir -p "$metadata_dir" "$archive_dir" "$log_dir"

base_url="https://huggingface.co/datasets/$REPO_ID/resolve/$REVISION"

download_file() {
  local url="$1"
  local final_path="$2"
  local log_path="$3"
  local part_path="${final_path}.part"

  if [[ -s "$final_path" ]]; then
    return 0
  fi

  curl --http1.1 \
    --silent \
    --show-error \
    --location \
    --fail \
    --retry 100 \
    --retry-all-errors \
    --retry-delay 10 \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 300 \
    --continue-at - \
    --output "$part_path" \
    "$url" >"$log_path" 2>&1
  mv "$part_path" "$final_path"
}

export -f download_file

download_metadata() {
  download_file \
    "$base_url/data/train/CSL_News_Labels.json" \
    "$metadata_dir/CSL_News_Labels.json" \
    "$log_dir/metadata_labels_json.log"
  download_file \
    "$base_url/data/train/CSL_News_Labels.csv" \
    "$metadata_dir/CSL_News_Labels.csv" \
    "$log_dir/metadata_labels_csv.log"

  curl --http1.1 --silent --show-error --location --fail --retry 10 --retry-all-errors \
    --output "$metadata_dir/README.md" \
    "https://huggingface.co/datasets/$REPO_ID/raw/$REVISION/README.md" \
    >"$log_dir/metadata_readme.log" 2>&1
}

if ((metadata_only)); then
  download_metadata
  exit 0
fi

available_bytes=$(df --output=avail -B1 "$output_dir" | tail -n 1 | tr -d ' ')
selected_count=$((end - start + 1))
if ((start == 1 && end == ARCHIVE_COUNT)); then
  required_bytes=$FULL_DATASET_BYTES
else
  required_bytes=$((selected_count * 2150000000))
fi

if ((available_bytes < required_bytes + reserve_bytes)); then
  echo "Insufficient capacity: available=$available_bytes required=$required_bytes reserve=$reserve_bytes" >&2
  exit 1
fi

printf 'repo=%s\nrevision=%s\nrange=%03d-%03d\nworkers=%d\nstarted_at=%s\n' \
  "$REPO_ID" "$REVISION" "$start" "$end" "$workers" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"$output_dir/DOWNLOAD_STATE"

download_archive() {
  local archive_id
  local current_available
  archive_id=$(printf '%03d' "$1")
  current_available=$(df --output=avail -B1 "$archive_dir" | tail -n 1 | tr -d ' ')
  if ((current_available < reserve_bytes)); then
    echo "Stopping before archive_${archive_id}: free space $current_available is below reserve $reserve_bytes" >&2
    return 255
  fi
  download_file \
    "$base_url/archive_${archive_id}.zip" \
    "$archive_dir/archive_${archive_id}.zip" \
    "$log_dir/archive_${archive_id}.log"
}

export base_url archive_dir log_dir reserve_bytes
export -f download_archive

seq "$start" "$end" | xargs -P "$workers" -n 1 bash -c 'download_archive "$1"' _
download_metadata

printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$output_dir/DOWNLOAD_STATE"
