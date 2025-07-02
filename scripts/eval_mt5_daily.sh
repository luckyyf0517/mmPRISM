#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12345 \
    run_peft.py \
    --config config/wavellm_mt5_daily.yaml \
    --batch-size 24 \
    --version "wavellm_mt5_daily_multimodal_0604_v5_eval" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_multimodal_0604_v5/last.ckpt"  \
    --dtype bf16 \
    --zero_stage 2 \
    --test 