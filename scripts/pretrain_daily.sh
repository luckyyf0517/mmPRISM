#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand_large_daily.yaml \
    --batch-size 32 \
    --max-epochs 20 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-large-daily-0525" \
    --resume-checkpoint "log/omnihand/omnihand-large-news-0523/last.ckpt" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
