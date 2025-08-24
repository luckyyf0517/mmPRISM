#!/bin/bash
deepspeed --include localhost:0,1 \
    run_peft.py \
    --config config/wavellm_phi3.yaml \
    --batch-size 8 \
    --max-epochs 5 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_phi3_0430_feature" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset