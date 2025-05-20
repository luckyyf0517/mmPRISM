#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm_mt5_pose.yaml \
    --batch-size 16 \
    --max-epochs 15 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_pose_0519_v1" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset
