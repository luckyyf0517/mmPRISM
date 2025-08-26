#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12345 \
    run_peft.py \
    --config config/wavellm_mt5_daily_multimodal.yaml \
    --batch-size 64 \
    --max-epochs 50 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_daily_multimodal_0825_01" \
    --dtype bf16 \
    --seed 42 \
    --zero_stage 2 \
    --resume-checkpoint "log/archived/wavellm_mt5_gt_pose_0523/last.ckpt" \
    --reset 
