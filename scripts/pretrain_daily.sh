#!/bin/bash
torchrun --nproc_per_node=2 \
    run_model.py \
    --config config/omnihand_base_daily.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "omnihand-base-daily-disc-0526" \
    --resume-checkpoint "log/omnihand/omnihand-base-news-disc-0526/last.ckpt" \
    --precision 32 \
    --reset
