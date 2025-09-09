#!/bin/bash
torchrun --nproc_per_node=2 \
    run_model.py \
    --config config/omnihand/omnihand_cubenet_daily.yaml \
    --batch-size 32 \
    --max-epochs 50 \
    --version "omnihand-cubenet-daily-0901" \
    # --resume-checkpoint "log/omnihand/omnihand-rtm-news-0627/last.ckpt" \
    --precision 32 \
    --reset
