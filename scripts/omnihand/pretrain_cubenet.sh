#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand/omnihand_cubenet_collected.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "omnihand-cubenet-collected-0830" \
    --precision 32 \
    --seed 42 \
    --reset 
    # --resume-checkpoint log/omnihand/omnihand-rtm-collected-0830/last.ckpt \
    # --test 