#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12345 \
    run_peft.py \
    --config config/wavellm_mt5_daily.yaml \
    --batch-size 64 \
    --max-epochs 30 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_daily_0617_no_pretrain" \
    --dtype bf16 \
    --seed 42 \
    --zero_stage 2 \
    --reset
    # --resume-checkpoint "log/archived/wavellm_mt5_gt_pose_0523/last.ckpt" 