#!/bin/bash

torchrun --nproc_per_node=1 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand_rtm_collected.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "omnihand_rtm_collected_exp" \
    --precision 32 \
    --seed 42