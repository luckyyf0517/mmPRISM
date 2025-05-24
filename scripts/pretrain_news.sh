#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand_large_news.yaml \
    --batch-size 32 \
    --max-epochs 15 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-large-news-0523" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
