#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12346 \
    run_model.py \
    --config config/omnihand/omnihand_mmhand_collected_demo.yaml \
    --batch-size 32 \
    --max-epochs 35 \
    --version "omnihand-mmhand-collected-demo-0831" \
    --precision 32 \
    --seed 42 \
    --resume-checkpoint log/omnihand/omnihand-mmhand-collected-0830/last.ckpt \
    --reset 
    # --test 
