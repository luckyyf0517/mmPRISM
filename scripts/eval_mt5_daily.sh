#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12345 \
    run_peft.py \
    --config config/wavellm_mt5_daily_pose.yaml \
    --batch-size 24 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_daily_pose_0519_v2_eval" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_pose_0519_v2/last.ckpt" \
    --dtype bf16 \
    --zero_stage 2 \
    --test