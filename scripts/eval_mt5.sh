#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm_mt5.yaml \
    --batch-size 24 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_0506_multi_modal" \
    --resume-checkpoint "log/peft_finetune/wavellm_mt5_0506_multi_modal/last.ckpt" \
    --dtype bf16 \
    --zero_stage 2 \
    --test