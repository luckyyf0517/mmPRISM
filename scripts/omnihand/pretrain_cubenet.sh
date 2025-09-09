#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand/omnihand_cubenet_collected_small.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "omnihand-cubenet-collected-small-0902-eval" \
    --precision 32 \
    --seed 42 \
    --resume-checkpoint log/omnihand/omnihand-cubenet-collected-small-0902/last.ckpt \
    --test
    # --reset 
