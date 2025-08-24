#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm_mt5_daily_features.yaml \
    --batch-size 24 \
    --version "wavellm_mt5_daily_features_0813_eval" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_daily_features_0813/last.ckpt"  \
    --dtype bf16 \
    --zero_stage 2 \
    --test 