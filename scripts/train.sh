#!/bin/bash
deepspeed --include localhost:0 \
    run_peft.py \
    --config config/wavellm_finetune.yaml \
    --batch-size 1 \
    --max-epochs 5 \
    --gradient-accumulation-steps 1 \
    --version "wavellm_finetune_0427_debug" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset