#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12345 \
    run_model.py \
    --config config/omnihand/omnihand_cubenet_collected_individual.yaml \
    --batch-size 32 \
    --max-epochs 100 \
    --version "omnihand-cubenet-collected-1108" \
    --precision 32 \
    --seed 42 \
    --reset 
