#!/bin/bash
deepspeed --include localhost:0,1 \
    run_model.py \
    --config config/omnihand_base_daily.yaml \
    --batch-size 32 \
    --max-epochs 20 \
    --gradient-accumulation-steps 1 \
    --version "omnihand-0509-daily" \
    --resume-checkpoint "log/omnihand/omnihand-0507/last.ckpt" \
    --dtype fp32 \
    --zero_stage 2 \
    --reset
