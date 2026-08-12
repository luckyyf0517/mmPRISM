#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_csl_news_annotation_worker.sh [options] [-- annotation-options]

Options:
  --config PATH        Annotation YAML configuration.
  --gpu ID|auto        Physical GPU index, or choose by free memory (default: auto).
  --min-free-mib N     Required free GPU memory before model load (default: 2048).
  --scheduled          Run the elastic lease-controlled worker instead of the
                       targeted/static annotation CLI.
  -h, --help            Show this help.

GPU utilization is intentionally not a gate. The operator approved sharing a
busy GPU as long as the free-memory threshold is satisfied.
EOF
}

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config="$project_root/configs/data/csl_news_rtmw3d_overnight.yaml"
gpu="auto"
min_free_mib=2048
scheduled=false
annotation_args=()

while (($#)); do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --min-free-mib)
      min_free_mib="$2"
      shift 2
      ;;
    --scheduled)
      scheduled=true
      shift
      ;;
    --)
      shift
      annotation_args=("$@")
      break
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

if ! [[ "$min_free_mib" =~ ^[1-9][0-9]*$ ]]; then
  echo "--min-free-mib must be a positive integer" >&2
  exit 2
fi

mapfile -t gpu_rows < <(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
)
selected=""
selected_free=-1
for row in "${gpu_rows[@]}"; do
  IFS=',' read -r index free_mib <<<"$row"
  index=${index//[[:space:]]/}
  free_mib=${free_mib//[[:space:]]/}
  if [[ "$gpu" != "auto" && "$index" != "$gpu" ]]; then
    continue
  fi
  if ((free_mib >= min_free_mib && free_mib > selected_free)); then
    selected="$index"
    selected_free="$free_mib"
  fi
done

if [[ -z "$selected" ]]; then
  echo "No requested GPU has at least ${min_free_mib} MiB free; retry later" >&2
  exit 75
fi

echo "Selected physical GPU ${selected} with ${selected_free} MiB free; utilization is not gated"
cd "$project_root"
export CUDA_VISIBLE_DEVICES="$selected"
# The pinned official checkpoint predates PyTorch's weights_only default change.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MMPRISM_CPU_THREADS="${MMPRISM_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$MMPRISM_CPU_THREADS"
export MKL_NUM_THREADS="$MMPRISM_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$MMPRISM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$MMPRISM_CPU_THREADS"
command="csl-news-annotate"
if [[ "$scheduled" == true ]]; then
  command="csl-news-annotate-scheduled"
fi
exec uv run --frozen --extra annotation mmprism "$command" \
  "$config" --project-root "$project_root" "${annotation_args[@]}"
