#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand_rtm_collected_temporal.yaml \
    --batch-size 8 \
    --max-epochs 30 \
    --version "omnihand_rtm_collected_temporal_exp" \
    --precision 32 \
    --seed 42