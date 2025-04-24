#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand.yaml \
    --batch-size 32 \
    --max-epochs 100 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-0423-vq" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
