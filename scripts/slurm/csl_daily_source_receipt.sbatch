#!/usr/bin/env bash
#SBATCH --job-name=mmprism-csldaily-receipt
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.err

# CSL-Daily source receipt: full-tree SHA-256 over the direct-preservation
# upload. Runs under Slurm because long non-Slurm processes on this node are
# reaped (gpu-direct-watchdog SIGTERMs non-allocation work).

set -euo pipefail

project_root=/home/yanyifan/mmPRISM
cd "$project_root"
export MMPRISM_DATA_ROOT=/mnt/gfs/yanyifan/mmPRISM

uv run --frozen mmprism csl-daily-source-receipt \
  --source-root /mnt/gfs/yanyifan/mmPRISM/external/csl_daily/csl_daily_original_20260812 \
  --receipt-root /mnt/gfs/yanyifan/mmPRISM/manifests/csl_daily/source_receipt_20260813 \
  --source-id csl_daily_original_20260812 \
  --legacy-split-root /mnt/gfs/yanyifan/mmPRISM/dataset/csl-daily \
  --stability-wait-seconds 90
