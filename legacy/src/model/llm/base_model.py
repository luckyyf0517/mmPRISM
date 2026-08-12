import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from transformers import PreTrainedModel
from einops.layers.torch import Rearrange

class WaveBaseModel(PreTrainedModel):
    """Base class for wave-enabled language models"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
            
    def process_wave_features(
        self,
        inputs_embeds: torch.FloatTensor,
        wave_embeds: torch.Tensor,
    ) -> torch.FloatTensor:
        """
        Simplified feature processing: directly concatenate text embeddings and modal features
        Args:
            inputs_embeds: Text embeddings [batch_size, seq_len, hidden_size]
            wave_embeds: Pose modality features [batch_size, num_frames, hidden_size]
        Returns:
            Concatenated embeddings [batch_size, seq_len+num_frames, hidden_size]
        """
        # Ensure both have the same hidden_size dimension
        assert inputs_embeds.shape[-1] == wave_embeds.shape[-1], \
            f"Feature dimension mismatch: text={inputs_embeds.shape[-1]}, pose={wave_embeds.shape[-1]}"
        
        # Directly concatenate along sequence dimension
        return torch.cat([inputs_embeds, wave_embeds], dim=1)

    def initialize_wave_modules(
        self,
        device: str,
        fix_llm: bool = True
    ) -> None:
        """
        Simplified initialization: only set parameter trainability
        """
        # Set whether to freeze LLM parameters
        if fix_llm:
            # Freeze model parameters
            for param in self.parameters():
                param.requires_grad = False
                
            # Input embeddings are trainable
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
                
            print("LLM parameters frozen, only input embeddings are trainable")
        else:
            # All parameters are trainable
            for param in self.parameters():
                param.requires_grad = True
                
            print("All LLM parameters are trainable")
