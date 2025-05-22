"""
Extracts and saves LLM parameters from a pre-trained checkpoint.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from termcolor import colored
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from src.utils.io import load_yaml
from src.utils.tools import instantiate_from_config
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from easydict import EasyDict as edict


if __name__ == "__main__":
    args = edict()
    args.config = 'config/wavellm_mt5_news_pose.yaml'
    args.checkpoint = 'log/peft_finetune/wavellm_mt5_pose_0515_continue/last.ckpt'

    # Load model from checkpoint
    model_cfg = load_yaml(args.config).model_cfg
    model_cfg.params.cfg.training.batch_size = 1
    model = instantiate_from_config(model_cfg)
    model = load_state_dict_from_zero_checkpoint(model, args.checkpoint)

    llm = model.model
    torch.save(llm.cpu().state_dict(), 'huggingface/mt5-pretrained-news-pose/pytorch_model.bin', _use_new_zipfile_serialization=True)
