import sys
sys.path.append(".")

from typing import Dict, Sequence, List
from src.model.llm.conversation import conv_phi3 as default_conversation

IGNORE_INDEX = -100

def insert_wave_tokens(
    sources: Sequence[str],
    wave_indicator: str = "<wave>",
    default_wave_patch_token: str = "<wave_patch>",
    wave_token_len: int = 248,
    use_wave_markers: bool = True,
    wave_start_token: str = "<wave_bos>",
    wave_end_token: str = "<wave_eos>"
) -> Dict:
    """
    Process and replace wave indicators with appropriate tokens in the input sources
    """
    for source in sources:
        for sentence in source:
            if sentence["value"] is None:
                continue
                
            token_sequence = default_wave_patch_token * wave_token_len
            if use_wave_markers:
                token_sequence = f"{wave_start_token}{token_sequence}{wave_end_token}"
            
            sentence["value"] = sentence["value"].replace(wave_indicator, token_sequence)
    return sources

def prepare_conversation_data(sources: Sequence[Dict], tokenizer) -> Dict:
    """
    Prepare and tokenize conversation data for model input
    Input: [<prompt>, <question>, <wave>, <answer>]
    Labels: [<masked_prompt>, <masked_wave>, <masked_question>, <answer>]
    """
    conv = default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    conversations = _build_conversations(sources, conv, roles)
    return _tokenize_and_mask(conversations, tokenizer, conv)

def create_conversation(questions: List[str], answers: List[str]) -> List[Dict]:
    """
    Create a formatted conversation pair from question and answer
    """
    if isinstance(questions, str):
        questions = [questions] * len(answers)
    else: 
        assert len(questions) == len(answers), "Questions and answers must have the same length"
    
    conversations = []
    for question, answer in zip(questions, answers):
        conversations.append([{
            "from": "human",
            "value": f'{question}\n<wave>'
        }, {
            "from": "gpt",
            "value": answer
        }])
    return conversations

def _build_conversations(sources, conv, roles):
    """Helper function to build conversation prompts"""
    conversations = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"Role mismatch at index {j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    return conversations

def _tokenize_and_mask(conversations, tokenizer, conv):
    """Helper function to tokenize and mask conversation data"""
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets
    _apply_target_masks(conversations, targets, tokenizer, conv)
    
    return {
        "input_ids": input_ids,
        "labels": targets,
    }

def _apply_target_masks(conversations, targets, tokenizer, conv):
    """Helper function to apply masks to target sequences"""
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


def prepare_simple_data(
    sources: Sequence[Dict], 
    tokenizer,
    prefix: str = "Translate sign language video to English: "
) -> Dict:
    """
    Prepare simple input format for encoder-decoder models like MT5
    Input: [<question>, <wave>]
    Labels: [<answer>]
    """
    # Prepare input prompts with wave tokens
    input_texts = []
    target_texts = []
    
    for source in sources:
        # Extract question (with wave token) and answer
        question = source[0]["value"]  # Contains <wave> token
        answer = source[1]["value"]    # Target caption
        
        # Combine prefix and question
        input_text = prefix + question
        
        input_texts.append(input_text)
        target_texts.append(answer)
    
    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    
    # Tokenize targets
    targets = tokenizer(
        target_texts,
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids
    
    # Replace padding token id with -100 for labels
    targets[targets == tokenizer.pad_token_id] = -IGNORE_INDEX
    
    return {
        "input_ids": inputs,
        "labels": targets
    }
