#!/bin/bash
torchrun --nproc_per_node=2 \
    run_model.py \
    --config config/omnihand_rtm_daily.yaml \
    --batch-size 32 \
    --max-epochs 50 \
    --version "omnihand-rtm-daily-disc-0601" \
    --resume-checkpoint "log/omnihand/omnihand-rtm-news-disc-0529/last.ckpt" \
    --precision 32 \
    --reset
