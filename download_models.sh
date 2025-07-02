#!/bin/bash

mkdir -p pretrained_models/simcse
mkdir -p pretrained_models/sbert

echo "Downloading SimCSE model..."
mkdir -p pretrained_models/simcse
wget https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext/resolve/main/config.json -P pretrained_models/simcse/
wget https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin -P pretrained_models/simcse/
wget https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext/resolve/main/tokenizer.json -P pretrained_models/simcse/
wget https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext/resolve/main/tokenizer_config.json -P pretrained_models/simcse/
wget https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext/resolve/main/vocab.txt -P pretrained_models/simcse/
wget https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext/resolve/main/special_tokens_map.json -P pretrained_models/simcse/

# echo "Downloading SBERT model..."
# mkdir -p pretrained_models/sbert
# wget https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/config.json -P pretrained_models/sbert/
# wget https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/pytorch_model.bin -P pretrained_models/sbert/
# wget https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/tokenizer.json -P pretrained_models/sbert/
# wget https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/tokenizer_config.json -P pretrained_models/sbert/
# wget https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/vocab.txt -P pretrained_models/sbert/
# wget https://huggingface.co/shibing624/text2vec-base-chinese/resolve/main/special_tokens_map.json -P pretrained_models/sbert/

echo "All models have been downloaded to pretrained_models/" 