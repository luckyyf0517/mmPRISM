import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import deepspeed
from omegaconf import OmegaConf
from src.model.llm.model_factory import ModelFactory
from src.model.llm.phi3_model import Phi3ModelWrapper
from src.model.llm.mt5_model import MT5ModelWrapper

def create_config():
    """Create model configuration as a dictionary"""
    config = {
        'model_type': 'phi3',  # or 'mt5'
        'model_path': 'huggingface/Phi-3-mini-4k-instruct',  # replace with actual model path
        'model_max_length': 512,
        'mm_input_dim': 768,  # match the hidden size of the language model
        'wave_start_token': '<|wave_start|>',
        'wave_end_token': '<|wave_end|>',
        'wave_patch_token': '<|wave_patch|>',
        'mm_use_wave_start_end': True
    }
    return OmegaConf.create(config)

def create_dummy_data(batch_size=2, num_patches=10, hidden_dim=768):
    """Create dummy wave features and text for testing"""
    wave_features = torch.randn(batch_size, num_patches, hidden_dim)
    # Use patch token as placeholder
    placeholder = '<|wave_patch|>' * num_patches
    captions = [f"Describe this wave pattern: <|wave_start|>{placeholder}<|wave_end|>."] * batch_size
    return {
        "wave_features": wave_features,
        "caption": captions
    }

def test_model(model, dummy_data, device):
    """Test model functionality"""
    print("Testing model...")
    
    # Move data to device
    wave_features = dummy_data["wave_features"].to(device)
    captions = dummy_data["caption"]
    
    # Print input text
    print("\nInput text:")
    for i, text in enumerate(captions):
        print(f"Sample {i+1}: {text}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    # Get tokenized input
    tokenized = model.tokenizer(captions, return_tensors="pt")
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    
    # Test forward pass
    outputs = model(
        input_wave_embeds=wave_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=None,
        labels=None
    )
    print(f"Forward pass successful! Output shape: {outputs.logits.shape}")
    
    # Generate text
    print("\nGenerating text...")
    generated_ids = model.generate(
        input_wave_embeds=wave_features,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50,
        num_beams=3
    )
    
    return outputs, generated_ids

def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    
    # Create configuration
    config = create_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = ModelFactory.create_model(config.model_type, config)
    model = model.to(device)
    
    # Create dummy data
    dummy_data = create_dummy_data()
    
    # Test model
    outputs, generated_ids = test_model(model, dummy_data, device)
    
    # Print outputs
    print("\nModel outputs:")
    print(f"Loss: {outputs.get('loss')}")
    print(f"Logits shape: {outputs['logits'].shape}")
    if outputs.get('decoder_hidden_states') is not None:
        print(f"Decoder hidden states shape: {outputs['decoder_hidden_states'][-1].shape}")
    
    # Decode generated text
    generated_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("\nGenerated text samples:")
    for i, text in enumerate(generated_text[:2]):  # Print first 2 samples
        print(f"Sample {i+1}: {text}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main() 