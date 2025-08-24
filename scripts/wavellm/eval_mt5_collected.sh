#!/bin/bash
deepspeed --include localhost:0 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm_mt5_collected.yaml \
    --batch-size 24 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_collected_0702_v0_eval" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_0702_gt_augmentation/last.ckpt" \
    --dtype bf16 \
    --zero_stage 2 \
    --test