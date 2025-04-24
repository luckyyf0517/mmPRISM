#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/peft_finetune.yaml \
    --batch-size 8 \
    --gradient-accumulation-steps 8 \
    --version "peft_finetune" \
    --dtype bf16 \
    --zero_stage 2 \
    --max-epochs 10 \
    --reset
