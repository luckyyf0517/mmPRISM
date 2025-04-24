#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm_finetune.yaml \
    --batch-size 32 \
    --max-epochs 5 \
    --gradient-accumulation-steps 1 \
    --version "wavellm_finetune" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset
