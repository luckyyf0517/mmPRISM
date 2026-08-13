#!/usr/bin/env bash
#SBATCH --job-name=mmprism-csldaily-pose-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.err

# CSL-Daily RTMW3D pilot annotation. Direct (non-Slurm) GPU processes are
# reaped by gpu-direct-watchdog on this node, so annotation must run inside a
# Slurm allocation. Device in the versioned config stays cuda:0: Slurm maps
# the allocated GPU to index 0 via CUDA_VISIBLE_DEVICES.

set -euo pipefail

project_root=/home/yanyifan/mmPRISM
config="${MMPRISM_CSL_DAILY_ANNOTATION_CONFIG:-$project_root/configs/data/csl_daily_rtmw3d.yaml}"
max_sequences="${MMPRISM_CSL_DAILY_PILOT_MAX_SEQUENCES:-105}"

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "This script must run inside a Slurm GPU allocation" >&2
  exit 2
fi

mkdir -p /mnt/gfs/yanyifan/mmPRISM/logs/slurm
cd "$project_root"
export MMPRISM_DATA_ROOT="${MMPRISM_DATA_ROOT:-/mnt/gfs/yanyifan/mmPRISM}"
export MMPRISM_CPU_THREADS="${MMPRISM_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$MMPRISM_CPU_THREADS"
export MKL_NUM_THREADS="$MMPRISM_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$MMPRISM_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$MMPRISM_CPU_THREADS"

echo "job_id=$SLURM_JOB_ID allocated_gpus=$CUDA_VISIBLE_DEVICES config=$config max_sequences=$max_sequences"
time uv run --frozen mmprism csl-daily-annotate \
  "$config" \
  --project-root "$project_root" \
  --max-sequences "$max_sequences"
