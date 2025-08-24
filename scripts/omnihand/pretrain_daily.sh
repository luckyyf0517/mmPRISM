#!/bin/bash
torchrun --nproc_per_node=2 \
    run_model.py \
    --config config/omnihand_rtm_daily.yaml \
    --batch-size 32 \
    --max-epochs 50 \
    --version "omnihand-rtm-daily-0627" \
    --resume-checkpoint "log/omnihand/omnihand-rtm-news-0627/last.ckpt" \
    --precision 32 \
    # --reset
