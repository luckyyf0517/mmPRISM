#!/bin/bash
torchrun --nproc_per_node=2 \
    --master_port 12346 \
    run_model.py \
    --config config/omnihand/omnihand_rtm_collected_exp_cubenet_100.yaml \
    --batch-size 32 \
    --max-epochs 30 \
    --version "omnihand-rtm-collected-cubenet-100-0815" \
    --precision 32 \
    --resume-checkpoint log/omnihand/omnihand-rtm-collected-cubenet-100-0815/last.ckpt \
    --test
    # --reset
