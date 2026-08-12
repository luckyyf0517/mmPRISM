#!/usr/bin/env bash
#SBATCH --job-name=mmprism-cslnews-pose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.err

set -euo pipefail

project_root=/home/yanyifan/mmPRISM
config="${MMPRISM_CSL_NEWS_ANNOTATION_CONFIG:-$project_root/configs/data/csl_news_rtmw3d_overnight.yaml}"
registry="${MMPRISM_CSL_NEWS_INTEGRITY_REGISTRY:-/mnt/gfs/yanyifan/mmPRISM/manifests/csl_news/source_integrity_v2/registry.json}"
worker_count="${MMPRISM_CSL_NEWS_WORKERS_PER_GPU:-2}"

if ! [[ "$worker_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "MMPRISM_CSL_NEWS_WORKERS_PER_GPU must be a positive integer" >&2
  exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "This script must run inside a Slurm GPU allocation" >&2
  exit 2
fi

mkdir -p /mnt/gfs/yanyifan/mmPRISM/logs/slurm
cd "$project_root"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MMPRISM_CPU_THREADS="${MMPRISM_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$MMPRISM_CPU_THREADS"
export MKL_NUM_THREADS="$MMPRISM_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$MMPRISM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$MMPRISM_CPU_THREADS"

echo "job_id=$SLURM_JOB_ID allocated_gpus=$CUDA_VISIBLE_DEVICES workers=$worker_count"
pids=()
for worker_index in $(seq 1 "$worker_count"); do
  worker_id="slurm-${SLURM_JOB_ID}-${worker_index}"
  uv run --frozen --extra annotation mmprism csl-news-annotate-scheduled \
    "$config" \
    --project-root "$project_root" \
    --integrity-registry "$registry" \
    --worker-id "$worker_id" &
  pids+=("$!")
done

terminate_workers() {
  trap - TERM INT
  kill -TERM "${pids[@]}" 2>/dev/null || true
  wait "${pids[@]}" || true
}
trap terminate_workers TERM INT

wait "${pids[@]}"
