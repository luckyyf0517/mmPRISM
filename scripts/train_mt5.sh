#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm_mt5.yaml \
    --batch-size 24 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_0430_pose" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset