#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm_mt5_daily.yaml \
    --batch-size 64 \
    --max-epochs 60 \
    --gradient-accumulation-steps 1 \
    --version "wavellm_mt5_daily_gt_pose_0521_v1.2" \
    --dtype bf16 \
    --zero_stage 2 \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_gt_pose_0521_v0/last.ckpt" \
    --reset