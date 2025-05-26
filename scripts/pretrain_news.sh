#!/bin/bash
torchrun --nproc_per_node=2 \
    run_model.py \
    --config config/omnihand_base_news.yaml \
    --batch-size 32 \
    --max-epochs 15 \
    --version "omnihand-base-news-disc-0526" \
    --precision 32 \
    --reset
