#!/usr/bin/env bash
#SBATCH --job-name=mmprism-csldaily-b64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.err

# Read-only batch benchmark. Submit two copies with different
# MMPRISM_CSL_DAILY_BENCHMARK_SEQUENCES values to use two Slurm GPUs.

set -euo pipefail

project_root=/home/yanyifan/mmPRISM
config="$project_root/configs/data/csl_daily_rtmw3d_v2.yaml"
report_root=/mnt/gfs/yanyifan/mmPRISM/interim/csl_daily/pose_annotation/rtmw3d_l_794dbc78_v2/benchmarks
sequences="${MMPRISM_CSL_DAILY_BENCHMARK_SEQUENCES:?set comma-separated benchmark sequence IDs}"
max_frames="${MMPRISM_CSL_DAILY_BENCHMARK_MAX_FRAMES:-0}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "This script must run inside a Slurm GPU allocation" >&2
  exit 2
fi

mkdir -p "$report_root" /mnt/gfs/yanyifan/mmPRISM/logs/slurm
cd "$project_root"
export MMPRISM_DATA_ROOT="${MMPRISM_DATA_ROOT:-/mnt/gfs/yanyifan/mmPRISM}"
export MMPRISM_CPU_THREADS="${MMPRISM_CPU_THREADS:-8}"
export OMP_NUM_THREADS="$MMPRISM_CPU_THREADS"
export MKL_NUM_THREADS="$MMPRISM_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$MMPRISM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$MMPRISM_CPU_THREADS"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

args=()
IFS=',' read -r -a sequence_array <<< "$sequences"
for sequence in "${sequence_array[@]}"; do
  args+=(--sequence-id "$sequence")
done
if [[ "$max_frames" != "0" ]]; then
  args+=(--max-frames-per-sequence "$max_frames")
fi

report="$report_root/batch64_${SLURM_JOB_ID}.json"
echo "job_id=$SLURM_JOB_ID allocated_gpus=$CUDA_VISIBLE_DEVICES sequences=$sequences report=$report"
time uv run --frozen --extra annotation python scripts/benchmark_csl_daily_annotation_batch.py \
  "$config" --project-root "$project_root" --batch-size 64 --output "$report" "${args[@]}"
