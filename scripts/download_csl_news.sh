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
  --engine NAME       Download engine: curl or aria2 (default: curl).
  --workers N         Concurrent archives (default: curl=16, aria2=4).
  --connections N     Connections per file for aria2 (default: 8).
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
engine="${MMPRISM_DOWNLOAD_ENGINE:-curl}"
workers=""
connections_per_file=8
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
    --engine)
      engine="$2"
      shift 2
      ;;
    --workers)
      workers="$2"
      shift 2
      ;;
    --connections)
      connections_per_file="$2"
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

case "$engine" in
  curl|aria2) ;;
  *)
    echo "Download engine must be curl or aria2: $engine" >&2
    exit 2
    ;;
esac

if [[ -z "$workers" ]]; then
  if [[ "$engine" == "aria2" ]]; then
    workers=4
  else
    workers=16
  fi
fi

if ((start < 1 || end > ARCHIVE_COUNT || start > end)); then
  echo "Archive range must satisfy 1 <= start <= end <= $ARCHIVE_COUNT" >&2
  exit 2
fi
if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "Worker count must be positive" >&2
  exit 2
fi
if ! [[ "$connections_per_file" =~ ^[1-9][0-9]*$ ]]; then
  echo "Connections per file must be positive" >&2
  exit 2
fi

aria2_bin="${MMPRISM_ARIA2_BIN:-}"
aria2_library_path="${MMPRISM_ARIA2_LIBRARY_PATH:-}"
if [[ "$engine" == "aria2" ]]; then
  if [[ -z "$aria2_bin" ]]; then
    aria2_bin=$(command -v aria2c || true)
  fi
  if [[ -z "$aria2_bin" || ! -x "$aria2_bin" ]]; then
    echo "aria2 engine requested but aria2c is unavailable; set MMPRISM_ARIA2_BIN" >&2
    exit 2
  fi
fi

metadata_dir="$output_dir/metadata"
archive_dir="$output_dir/rgb_archives"
log_dir="$output_dir/logs"
mkdir -p "$metadata_dir" "$archive_dir" "$log_dir"

base_url="https://huggingface.co/datasets/$REPO_ID/resolve/$REVISION"

run_aria2() {
  local url="$1"
  local part_path="$2"
  local log_path="$3"
  local -a command=(
    "$aria2_bin"
    --continue=true
    --max-connection-per-server="$connections_per_file"
    --split="$connections_per_file"
    --min-split-size=1M
    --piece-length=1M
    --file-allocation=none
    --allow-overwrite=true
    --auto-file-renaming=false
    --max-tries=0
    --retry-wait=10
    --connect-timeout=30
    --timeout=300
    --lowest-speed-limit=0
    --summary-interval=30
    --dir="$(dirname "$part_path")"
    --out="$(basename "$part_path")"
    "$url"
  )

  if [[ -n "$aria2_library_path" ]]; then
    LD_LIBRARY_PATH="$aria2_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "${command[@]}" >"$log_path" 2>&1
  else
    "${command[@]}" >"$log_path" 2>&1
  fi
}

download_file() {
  local url="$1"
  local final_path="$2"
  local log_path="$3"
  local part_path="${final_path}.part"

  if [[ -s "$final_path" ]]; then
    return 0
  fi

  if [[ "$engine" == "aria2" ]]; then
    run_aria2 "$url" "$part_path" "$log_path"
  else
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
  fi
  mv "$part_path" "$final_path"
}

export -f run_aria2 download_file

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

printf 'repo=%s\nrevision=%s\nrange=%03d-%03d\nengine=%s\nworkers=%d\nconnections_per_file=%d\nstarted_at=%s\n' \
  "$REPO_ID" "$REVISION" "$start" "$end" "$engine" "$workers" "$connections_per_file" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
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

export base_url archive_dir log_dir reserve_bytes engine
export aria2_bin aria2_library_path connections_per_file
export -f download_archive

seq "$start" "$end" | xargs -P "$workers" -n 1 bash -c 'download_archive "$1"' _
download_metadata

printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$output_dir/DOWNLOAD_STATE"
