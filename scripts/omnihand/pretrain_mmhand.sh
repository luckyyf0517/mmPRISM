#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12346 \
    run_model.py \
    --config config/omnihand/omnihand_mmhand_collected.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "omnihand-mmhand-collected-0830" \
    --precision 32 \
    --seed 42 \
    --reset 
    # --resume-checkpoint log/omnihand/omnihand-mmhand-collected-0830/last.ckpt \
    # --test 
