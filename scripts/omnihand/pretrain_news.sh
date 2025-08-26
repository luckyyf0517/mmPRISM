#!/bin/bash
torchrun --nproc_per_node=2 \
    --master_port 12345 \
    run_model.py \
    --config config/omnihand/omnihand_rtm_news.yaml \
    --batch-size 32 \
    --max-epochs 100 \
    --version "omnihand-rtm-news-0701" \
    --precision 32 \
    --reset
