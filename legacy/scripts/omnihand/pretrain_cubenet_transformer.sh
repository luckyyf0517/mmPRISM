#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12348 \
    run_model.py \
    --config config/omnihand/omnihand_cubenet_collected_transformer_individual.yaml \
    --batch-size 8 \
    --max-epochs 30 \
    --version "omnihand_cubenet_collected_transformer_individual" \
    --precision 32 \
    --seed 42 \
    --reset
