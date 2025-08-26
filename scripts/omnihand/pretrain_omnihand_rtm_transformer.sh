#!/bin/bash

torchrun --nproc_per_node=2 \
    --master_port 12349 \
    run_model.py \
    --config config/omnihand/omnihand_rtm_collected_transformer.yaml \
    --batch-size 8 \
    --max-epochs 30 \
    --version "omnihand_rtm_collected_transformer_exp" \
    --precision 32 \
    --seed 42 \
    --resume-checkpoint "log/exp/omnihand_rtm_collected_transformer_exp/last.ckpt" \
    --test