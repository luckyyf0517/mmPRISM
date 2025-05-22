#!/bin/bash
deepspeed --include localhost:0,1 \
    --master_port 12347 \
    run_alignment.py \
    --config config/alignment.yaml \
    --batch-size 32 \
    --max-epochs 10 \
    --version "alignment_pred_0521_v1" \
    --dtype bf16 \
    --zero_stage 2