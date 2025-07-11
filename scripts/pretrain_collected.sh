#!/bin/bash
torchrun --nproc_per_node=2 \
    --master_port 12350 \
    run_model.py \
    --config config/omnihand_mmhand_collected.yaml \
    --batch-size 32 \
    --max-epochs 100 \
    --version "omnihand-mmhand-collected-0707" \
    --precision 32 \
    --reset
    # --test
