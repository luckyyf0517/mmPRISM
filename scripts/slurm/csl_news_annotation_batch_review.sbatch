#!/usr/bin/env bash
#SBATCH --job-name=mmprism-cslnews-batch-review
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.out
#SBATCH --error=/mnt/gfs/yanyifan/mmPRISM/logs/slurm/%x-%j.err

set -euo pipefail

project_root=/home/yanyifan/mmPRISM
config="$project_root/configs/data/csl_news_rtmw3d_overnight.yaml"
video="${MMPRISM_CSL_NEWS_BATCH_REVIEW_VIDEO:-/mnt/gfs/yanyifan/mmPRISM/cache/csl_news_annotation/rtmw3d_l_794dbc78_v1/videos/archive_051/0aad8c7516249aaad32cfe86--Common-Concerns_20200229_18287-18537_313410.mp4}"
output_dir=/mnt/gfs/yanyifan/mmPRISM/interim/csl_news/pose_annotation/rtmw3d_l_794dbc78_v1/reviews/batch_1_16_64_${SLURM_JOB_ID}

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "This script must run inside a Slurm GPU allocation" >&2
  exit 2
fi

mkdir -p "$output_dir" /mnt/gfs/yanyifan/mmPRISM/logs/slurm
cd "$project_root"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export MMPRISM_CPU_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "job_id=$SLURM_JOB_ID allocated_gpus=$CUDA_VISIBLE_DEVICES video=$video output_dir=$output_dir"
uv run --frozen --extra annotation --extra visualization python scripts/review_csl_news_annotation_batch.py \
  "$config" "$video" \
  --project-root "$project_root" \
  --output-dir "$output_dir"
