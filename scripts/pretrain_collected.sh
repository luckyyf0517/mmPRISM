#!/bin/bash
torchrun --nproc_per_node=2 \
    --master_port 12345 \
    run_model.py \
    --config config/omnihand_rtm_collected.yaml \
    --batch-size 32 \
    --max-epochs 25 \
    --version "omnihand-rtm-collected-no-disc-0813" \
    --precision 32 \
    --reset
