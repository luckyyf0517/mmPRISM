#!/bin/bash

cfg_name="wavellm_mt5_daily_xy"
exp_name="wavellm_mt5_daily_0827_xy"

deepspeed --include localhost:0,1 \
    --master_port 12346 \
    run_peft.py \
    --config config/wavellm/${cfg_name}.yaml \
    --batch-size 24 \
    --version "${exp_name}_eval" \
    --resume-checkpoint "log/exp/${exp_name}/last.ckpt"  \
    --dtype bf16 \
    --zero_stage 2 \
    --test 

python run_evaluation.py --results_dir log/exp/${exp_name}/last.ckpt/evaluation