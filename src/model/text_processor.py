import torch
from typing import Dict, Sequence
from src.model.llm.conversation import conv_phi3 as default_conversation

IGNORE_INDEX = -100

def preprocess_multimodal_wave(
    sources: Sequence[str],
    wave_indicator: str = "<wave>",
    default_wave_patch_token: str = "<wave_patch>",
    wave_token_len: int = 248,
    mm_use_wave_start_end: bool = True,
    default_wave_start_token: str = "<wave_bos>",
    default_wave_end_token: str = "<wave_eos>"
) -> Dict:
    for source in sources:
        for sentence in source:
            replace_token = default_wave_patch_token * wave_token_len 
            if mm_use_wave_start_end:
                replace_token = default_wave_start_token + replace_token + default_wave_end_token
            if sentence["value"] is not None:
                sentence["value"] = sentence["value"].replace(wave_indicator, replace_token)
    return sources

def preprocess(sources, tokenizer):
    conv = default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets
    sep = conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2])) # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len + 1
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch precess_v3: {cur_len} vs. {total_len}."
                    f" (ignored)")
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def format_conversation(question: str = None, answer: str = None) -> list:
    
    return [{
        "from": "human",
        "value": f"<wave>\n{question}"
    }, {
        "from": "gpt",
        "value": answer
    }]