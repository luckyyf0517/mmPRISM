import torch.nn as nn
from typing import Dict, Any, Type
from src.model.llm.phi3_model import Phi3ForCausalLM
from src.model.llm.mt5_model import MT5ForConditionalGeneration
from easydict import EasyDict
import os
import torch
from transformers import AutoConfig, AutoTokenizer, MT5Tokenizer
from termcolor import colored

class ModelFactory:
    """Factory class for creating different types of models and tokenizers"""
    
    MODEL_TYPES = {
        'phi3': Phi3ForCausalLM,
        'mt5': MT5ForConditionalGeneration,
    }
    
    @staticmethod
    def create_model(model_type, model_config):
        """
        Create a model instance
        Args:
            model_type: Model type, e.g., "mt5"
            model_config: Model configuration
        Returns:
            Model instance
        """
        if model_type.lower() == "mt5":
            # Get the pre-trained model path
            model_path = model_config.get("model_path", "google/mt5-base")
            
            # Load the pre-trained model
            if os.path.exists(model_path):
                print(f"Loading model from local: {model_path}")
                model = MT5ForConditionalGeneration.from_pretrained(model_path)
            else:
                print(f"Loading model from Hugging Face: {model_path}")
                model = MT5ForConditionalGeneration.from_pretrained(model_path)
            
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_tokenizer(model_config):
        """
        Create a tokenizer instance
        Args:
            model_config: Model configuration
        Returns:
            Tokenizer instance
        """
        # Get the pre-trained model path
        model_path = model_config.get("model_path", "google/mt5-base")
        
        # Get the model's maximum length configuration
        model_max_length = model_config.get("model_max_length", 2048)
        
        # Load the tokenizer
        if "mt5" in model_path.lower():
            tokenizer = MT5Tokenizer.from_pretrained(
                model_path,
                model_max_length=model_max_length,
                legacy=False
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                model_max_length=model_max_length,
                padding_side="right",
                use_fast=False
            )
            
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
        
        # Get the model class
        model_class = cls.MODEL_TYPES[model_type]
        
        # First load the original config
        print(colored("[ModelFactory] Loading model config", "blue"))
        original_config = AutoConfig.from_pretrained(
            config.model_path,
            cache_dir=config.get('cache_dir'),
        )
        
        # Update the config with our custom settings
        print(colored("[ModelFactory] Updating config with custom settings", "blue"))
        custom_config = {
            'model_max_length': config.get('model_max_length', 2048),
            'use_cache': config.get('use_cache', True),  
        }
        for key, value in custom_config.items():
            if hasattr(original_config, key):
                print(colored(f"[ModelFactory] Overriding {key}: {getattr(original_config, key)} -> {value}", "yellow"))
                setattr(original_config, key, value)
            else:
                print(colored(f"[ModelFactory] Setting new attribute {key}: {value}", "yellow"))
                setattr(original_config, key, value)
        
        print(colored("[ModelFactory] Loading pretrained model", "blue"))
        model = model_class.from_pretrained(
            config.model_path,
            config=original_config,
            cache_dir=config.get('cache_dir'),
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=False,
        )
        
        # Set the model parameter type
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