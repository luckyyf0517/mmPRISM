#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand_base.yaml \
    --batch-size 32 \
    --max-epochs 10 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-0507-no-learnable-weights" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
