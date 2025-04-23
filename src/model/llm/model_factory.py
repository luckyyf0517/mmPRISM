import torch.nn as nn
from typing import Dict, Any, Type
from src.model.llm.phi3_model import Phi3ModelWrapper
from src.model.llm.mt5_model import MT5ModelWrapper

class ModelFactory:
    """Factory class for creating different models"""
    
    # Model type mapping
    MODEL_TYPES = {
        'phi3': Phi3ModelWrapper,
        'mt5': MT5ModelWrapper,
        # Add more model types here
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model based on model type and configuration
        
        Args:
            model_type: Model type, e.g., 'phi3', 'mt5', etc.
            config: Model configuration
            
        Returns:
            Created model instance
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}, supported types: {list(cls.MODEL_TYPES.keys())}")
        
        model_class = cls.MODEL_TYPES[model_type]
        model = model_class(config)
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        return model
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[nn.Module]):
        """
        Register a new model type
        
        Args:
            model_type: Model type name
            model_class: Model class
        """
        cls.MODEL_TYPES[model_type] = model_class 