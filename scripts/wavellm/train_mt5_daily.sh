#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12345 \
    run_peft.py \
    --config config/wavellm/wavellm_mt5_daily.yaml \
    --batch-size 64 \
    --max-epochs 50 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_daily_0827_aug" \
    --dtype bf16 \
    --seed 42 \
    --zero_stage 2 \
    --resume-checkpoint "log/archived/wavellm_mt5_daily_pred_pose_B21/last.ckpt" \
    --reset 
