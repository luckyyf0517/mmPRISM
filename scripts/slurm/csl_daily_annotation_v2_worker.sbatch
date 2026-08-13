#!/usr/bin/env bash
#SBATCH --job-name=mmprism-csldaily-pose-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.err

# One elastic CSL-Daily annotation_v2 worker. Submit additional copies as
# capacity permits; durable sequence leases prevent duplicate work. The queue
# is initialized paused, then explicitly resumed by the operator.

set -euo pipefail

project_root=/home/yanyifan/mmPRISM
config="${MMPRISM_CSL_DAILY_ANNOTATION_V2_CONFIG:-$project_root/configs/data/csl_daily_rtmw3d_v2.yaml}"

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

worker_id="${HOSTNAME}:slurm-${SLURM_JOB_ID}"
time uv run --frozen mmprism csl-daily-annotate-scheduled \
  "$config" \
  --project-root "$project_root" \
  --worker-id "$worker_id" \
  --poll-seconds 15
