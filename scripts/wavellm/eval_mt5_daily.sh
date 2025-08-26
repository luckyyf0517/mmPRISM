#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm/wavellm_mt5_daily_multimodal.yaml \
    --batch-size 24 \
    --version "wavellm_mt5_daily_multimodal_0825_eval" \
    --resume-checkpoint "log/old/peft_finetune/wavellm_mt5_daily_0702_gt_augmentation/last.ckpt"  \
    --dtype bf16 \
    --zero_stage 2 \
    --test 