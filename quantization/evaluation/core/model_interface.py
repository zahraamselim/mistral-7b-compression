"""Abstract base class for all model implementations."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)


class ModelInterface(ABC):
    """
    Abstract interface for all model implementations.
    
    This is the only place that knows about model internals.
    All benchmarks and analysis tools use this interface.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_path = None
        self.model_type = None
    
    @abstractmethod
    def load(self, model_path: str, **kwargs):
        """
        Load model and tokenizer.
        
        Args:
            model_path: Path or HuggingFace hub ID
            **kwargs: Implementation-specific arguments
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_loglikelihood(self, text: str, context: str = "") -> float:
        """
        Calculate log-likelihood of text given context.
        
        Critical for accuracy benchmarks.
        
        Args:
            text: Full text to evaluate
            context: Context prefix
            
        Returns:
            Log probability of continuation
        """
        pass
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            **kwargs: Additional forward pass arguments
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        pass
    
    def get_model(self):
        """Get underlying model object."""
        return self.model
    
    def get_tokenizer(self):
        """Get tokenizer."""
        return self.tokenizer
    
    def get_device(self):
        """Get device."""
        return self.device
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": str(self.device),
        }
        
        if hasattr(self.model, 'parameters'):
            param_size = sum(p.element_size() * p.numel() for p in self.model.parameters())
            info["size_gb"] = param_size / (1024**3)
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
        
        if torch.cuda.is_available() and 'cuda' in str(self.device):
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        return info
    
    def get_lm_eval_model(self):
        """
        Get lm-eval compatible model wrapper.
        
        Override if custom integration needed.
        
        Returns:
            HFLM instance or None
        """
        try:
            from lm_eval.models.huggingface import HFLM
            return HFLM(
                pretrained=self.model,
                tokenizer=self.tokenizer,
                batch_size=1
            )
        except Exception as e:
            logger.warning(f"Could not create HFLM wrapper: {e}")
            return None
    
    def supports_lm_eval(self) -> bool:
        """Check if model works with lm-eval."""
        return self.get_lm_eval_model() is not None


def create_model_interface(model_type: str = "huggingface") -> ModelInterface:
    """
    Factory function to create model interface.
    
    Args:
        model_type: Type of model interface
            - 'huggingface' or 'hf': Standard HuggingFace models
            - 'gptq': GPTQ 4-bit quantized
            - 'awq': AWQ 4-bit quantized
            - 'hqq': HQQ quantized (2/3/4/8-bit)
            - 'exllamav2' or 'exl2': ExLlamaV2 optimized GPTQ
        
    Returns:
        ModelInterface instance
        
    Examples:
        >>> model = create_model_interface('huggingface')
        >>> model.load('mistralai/Mistral-7B-Instruct-v0.1')
        
        >>> model = create_model_interface('exllamav2')
        >>> model.load('./mistral-7b-gptq-4bit')
    """
    model_type = model_type.lower()
    
    if model_type in ['huggingface', 'hf']:
        from models.huggingface import HuggingFaceModel
        return HuggingFaceModel()
    
    elif model_type == 'gptq':
        from models.gptq import GPTQModel
        return GPTQModel()
    
    elif model_type == 'awq':
        from models.awq import AWQModel
        return AWQModel()
    
    elif model_type == 'hqq':
        from models.hqq import HQQModel
        return HQQModel()
    
    elif model_type in ['exllamav2', 'exl2']:
        from models.exllamav2 import ExLlamaV2Model
        return ExLlamaV2Model()
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: huggingface, gptq, awq, hqq, exllamav2"
        )