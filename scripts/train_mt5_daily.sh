#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm_mt5_daily_multimodal.yaml \
    --batch-size 8 \
    --max-epochs 80 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_daily_multimodal_0603" \
    --dtype bf16 \
    --zero_stage 2 \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_pred_pose_rtm/last.ckpt" 
    # --reset