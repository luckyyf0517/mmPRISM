#!/bin/bash
deepspeed --include localhost:0,1 \
	--master_port 1234 \
    run_peft.py \
    --config config/wavellm_mt5_news_flow.yaml \
    --batch-size 16 \
    --max-epochs 15 \
    --gradient-accumulation-steps 8 \
    --version "wavellm_mt5_news_flow_0526" \
    --dtype bf16 \
    --zero_stage 2 \
    --reset
