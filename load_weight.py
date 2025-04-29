import torch

pretrained_model = "temp/csl_stage1_weight.pth"
pretrained_model_dict = torch.load(pretrained_model)['model']

pretrained_model_dict = {k.replace("mt5_model.", ""): v for k, v in pretrained_model_dict.items() if "mt5_model." in k}

torch.save(pretrained_model_dict, "huggingface/mt5-pretrained/pytorch_model.bin")