#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/mmwave2text.yaml \
    --batch-size 32 \
    --gradient-accumulation-steps 8 \
    --version "deepspeed_training" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset
