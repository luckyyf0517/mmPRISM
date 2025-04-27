import torch.nn as nn
from typing import Dict, Any, Type
from src.model.llm.phi3_model import Phi3ForCausalLM
from easydict import EasyDict
import os
import torch
from transformers import AutoConfig, AutoTokenizer
from termcolor import colored

class ModelFactory:
    """Factory class for creating different models"""
    
    MODEL_TYPES = {
        'phi3': Phi3ForCausalLM,
    }
    
    @classmethod
    def create_tokenizer(cls, config: EasyDict):
        """
        Create a tokenizer based on configuration
        
        Args:
            config: Configuration containing model_path and other settings
            
        Returns:
            Created tokenizer instance
        """
        print(colored(f"[ModelFactory] Creating tokenizer from: {config.model_path}", "blue"))
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            cache_dir=config.get('cache_dir'),
            model_max_length=config.get('model_max_length', 2048),
            padding_side="right",
            use_fast=True,
        )
        
        print(colored("[ModelFactory] Tokenizer creation completed", "green"))
        return tokenizer

    @classmethod
    def create_model(cls, model_type: str, config: EasyDict) -> nn.Module:
        """
        Create a model based on model type and configuration
        
        Args:
            model_type: Model type, e.g., 'phi3'
            config: Model configuration
            
        Returns:
            Created model instance
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}, supported types: {list(cls.MODEL_TYPES.keys())}")
        
        print(colored(f"[ModelFactory] Creating model of type: {model_type}", "green"))
        
        # Get model class
        model_class = cls.MODEL_TYPES[model_type]
        
        print(colored("[ModelFactory] Loading pretrained model", "blue"))
        model = model_class.from_pretrained(
            config.model_path,
            cache_dir=config.get('cache_dir'),
            torch_dtype=torch.bfloat16
        )

        # Disable model cache for training
        model.config.use_cache = False
        
        # Set model parameter type
        if config.get('torch_dtype'):
            print(colored(f"[ModelFactory] Converting model to dtype: {config.torch_dtype}", "blue"))
            model = model.to(dtype=config.torch_dtype)
        
        print(colored("[ModelFactory] Model creation completed successfully", "green"))
        return model

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[nn.Module]):
        """Register a new model type"""
        print(colored(f"[ModelFactory] Registering new model type: {model_type}", "yellow"))
        cls.MODEL_TYPES[model_type] = model_class 