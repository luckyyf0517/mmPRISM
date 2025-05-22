#!/bin/bash
deepspeed --include localhost:0,1 \
	--master_port 1234 \
    run_peft.py \
    --config config/wavellm_mt5_news_pose.yaml \
    --batch-size 16 \
    --max-epochs 5 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_gt_pose_0521_v0" \
    --dtype bf16 \
    --zero_stage 2 \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_pose_0515/last.ckpt" \
    --reset
