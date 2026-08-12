#!/bin/bash

cfg_name="wavellm_mt5_daily_xy"
exp_name="wavellm_mt5_daily_0827_xy"

deepspeed --include localhost:0,1 \
    --master_port 12345 \
    run_peft.py \
    --config config/wavellm/${cfg_name}.yaml \
    --batch-size 64 \
    --max-epochs 10 \
    --gradient-accumulation-steps 8 \
    --version "${exp_name}" \
    --dtype bf16 \
    --seed 42 \
    --zero_stage 2 \
    --reset \
    --resume-checkpoint "log/archived/wavellm_mt5_daily_0826/last.ckpt" 
