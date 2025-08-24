#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12348 \
    run_model.py \
    --config config/omnihand_tvan.yaml \
    --batch-size 8 \
    --max-epochs 30 \
    --version "omnihand_tvan_exp_5frames" \
    --precision 32 \
    --seed 42
    # --reset