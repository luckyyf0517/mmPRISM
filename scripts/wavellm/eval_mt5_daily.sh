#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm/wavellm_mt5_daily.yaml \
    --batch-size 24 \
    --version "wavellm_mt5_daily_0827_aug_eval" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_0827_aug/last.ckpt"  \
    --dtype bf16 \
    --zero_stage 2 \
    --test 