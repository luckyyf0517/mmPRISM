#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand_base_news_dual.yaml \
    --batch-size 32 \
    --max-epochs 15 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-news-dual-0522" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
