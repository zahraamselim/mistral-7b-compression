"""
Model Interface for LLM Evaluation Framework

Provides an abstract interface for model evaluation to ensure consistency
across different model implementations and quantization methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import torch
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


@dataclass
class ModelOutput:
    """Standardized output from model generation"""
    generated_ids: torch.Tensor
    generated_text: str
    attentions: Optional[List[torch.Tensor]] = None
    num_generated_tokens: int = 0
    latency_ms: float = 0.0


class ModelInterface(ABC):
    """
    Abstract interface for language models.
    
    All model implementations must inherit from this class and implement
    the required methods. This ensures consistent evaluation across different
    model types (e.g., quantized, full-precision, different architectures).
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the model.
        
        Args:
            model_path: Path to model weights or HuggingFace model ID
            device: Device to load model on ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = device
        self._model = None
        self._tokenizer = None
        self._model_type = None
    
    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer into memory"""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        return_attentions: bool = False
    ) -> ModelOutput:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            return_attentions: Whether to return attention weights
            
        Returns:
            ModelOutput containing generated text and metadata
        """
        pass
    
    @abstractmethod
    def get_perplexity(
        self,
        text: str,
        max_length: int = 512
    ) -> float:
        """
        Calculate perplexity on input text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Perplexity score
        """
        pass
    
    @abstractmethod
    def encode(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and return input tensors.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (None for no truncation)
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.
        
        Returns:
            Dictionary with model information (size, dtype, quantization, etc.)
        """
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB
        """
        pass
    
    @abstractmethod
    def clear_memory(self) -> None:
        """Clear GPU/CPU cache and force garbage collection"""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory"""
        pass
    
    @property
    def tokenizer(self):
        """Get tokenizer instance"""
        return self._tokenizer
    
    @property
    def model(self):
        """Get model instance"""
        return self._model
    
    @property
    def model_type(self):
        """Get model type (base or instruct)"""
        return self._model_type
    
    def __enter__(self):
        """Context manager entry"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload()