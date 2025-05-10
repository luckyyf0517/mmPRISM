#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm_mt5_daily_pose.yaml \
    --batch-size 24 \
    --max-epochs 20 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_0510_daily_pose" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset