#!/bin/bash
torchrun --nproc_per_node=2 \
    --master_port 12345 \
    run_model.py \
    --config config/omnihand_base_news.yaml \
    --batch-size 32 \
    --max-epochs 15 \
    --version "omnihand-base-news-0527" \
    --precision 32 \
    --reset
