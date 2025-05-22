#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand_base_daily_dual.yaml \
    --batch-size 32 \
    --max-epochs 20 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-0522-daily-dual" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
