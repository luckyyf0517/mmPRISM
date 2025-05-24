#!/bin/bash
deepspeed --include localhost:0,1 \
	--master_port 1234 \
    run_peft.py \
    --config config/wavellm_mt5_news.yaml \
    --batch-size 16 \
    --max-epochs 15 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_pred_pose_0523_v0" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset
