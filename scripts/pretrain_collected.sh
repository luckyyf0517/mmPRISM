#!/bin/bash
torchrun --nproc_per_node=2 \
    --master_port 12350 \
    run_model.py \
    --config config/omnihand_rtm_collected.yaml \
    --batch-size 32 \
    --max-epochs 300 \
    --version "omnihand-rtm-collected-0704-disc" \
    --precision 32 \
    --resume-checkpoint log/omnihand/omnihand-rtm-collected-0704/last.ckpt 
    # --test
    # --reset
