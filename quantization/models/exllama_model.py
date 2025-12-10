"""
ExLlamaV2 Model Implementation

Implements the ModelInterface for GPTQ models using ExLlamaV2 backend.
Focus: Fast inference for quantized models.
"""

import gc
import time
from typing import Dict, Any, List, Optional
import torch
from pathlib import Path
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

from models.model_interface import ModelInterface, GenerationConfig, ModelOutput


class ExLlamaModel(ModelInterface):
    """ExLlamaV2 model implementation for GPTQ quantized models"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        super().__init__(model_path, device)
        self._config = None
        self._cache = None
        self._generator = None
    
    def load(self):
        """Load model using ExLlamaV2"""
        print(f"Loading GPTQ model with ExLlamaV2: {self.model_path}")
        
        # Initialize config
        self._config = ExLlamaV2Config()
        self._config.model_dir = self.model_path
        self._config.prepare()
        
        # Load model
        self._model = ExLlamaV2(self._config)
        self._cache = ExLlamaV2Cache(self._model, lazy=True)
        self._model.load_autosplit(self._cache)
        
        # Load tokenizer
        self._tokenizer = ExLlamaV2Tokenizer(self._config)
        
        # Initialize generator
        self._generator = ExLlamaV2StreamingGenerator(
            self._model,
            self._cache,
            self._tokenizer
        )
        
        # Detect model type from model name
        model_name_lower = self.model_path.lower()
        if "instruct" in model_name_lower or "chat" in model_name_lower:
            self._model_type = "instruct"
        else:
            self._model_type = "base"
        
        print(f"Model loaded on: {self.device}")
        print(f"Model type detected: {self._model_type}")
    
    def generate(self, prompt: str, config: GenerationConfig, return_attentions: bool = False) -> ModelOutput:
        """
        Generate text from prompt.
        
        Note: ExLlamaV2 doesn't support attention extraction.
        return_attentions parameter is ignored.
        """
        if return_attentions:
            print("Warning: ExLlamaV2 does not support attention extraction")
        
        # Setup generation settings
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = config.temperature
        settings.top_p = config.top_p
        settings.top_k = config.top_k
        settings.token_repetition_penalty = 1.0
        
        # Reset cache
        self._cache.current_seq_len = 0
        
        # Encode prompt
        input_ids = self._tokenizer.encode(prompt)
        input_length = input_ids.shape[1]
        
        start_time = time.perf_counter()
        
        # Generate
        output_text = self._generator.generate_simple(
            prompt,
            settings,
            config.max_new_tokens,
            seed=42 if not config.do_sample else None
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract generated text (remove prompt)
        generated_text = output_text[len(prompt):]
        
        # Approximate token count
        generated_ids = self._tokenizer.encode(generated_text)
        num_tokens = generated_ids.shape[1]
        
        return ModelOutput(
            generated_ids=generated_ids,
            generated_text=generated_text,
            attentions=None,
            num_generated_tokens=num_tokens,
            latency_ms=latency_ms
        )
    
    def get_perplexity(self, text: str, max_length: int = 512) -> float:
        """Calculate perplexity on text"""
        self._cache.current_seq_len = 0
        
        input_ids = self._tokenizer.encode(text)
        if input_ids.shape[-1] > max_length:
            input_ids = input_ids[:, :max_length]
        
        if input_ids.shape[-1] < 2:
            return float('inf')
        
        # Forward pass
        logits = self._model.forward(input_ids, self._cache, input_mask=None)
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        
        # Calculate loss
        shift_logits = logits[:-1, :]
        shift_labels = input_ids[0, 1:].to(shift_logits.device)
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='mean'
        )
        
        return torch.exp(loss).item()
    
    def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Tokenize text"""
        input_ids = self._tokenizer.encode(text)
        if max_length and input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs"""
        return self._tokenizer.decode(token_ids)[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info"""
        return {
            "model_path": self.model_path,
            "model_type": self._model_type,
            "num_layers": self._config.num_hidden_layers,
            "num_attention_heads": self._config.num_attention_heads,
            "hidden_size": self._config.hidden_size,
            "vocab_size": self._config.vocab_size,
            "dtype": "gptq_4bit",
            "backend": "exllamav2"
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if torch.cuda.is_available():
            return {
                "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2)
            }
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    
    def clear_memory(self):
        """Clear memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def unload(self):
        """Unload model from memory"""
        if self._generator:
            del self._generator
        if self._cache:
            del self._cache
        if self._model:
            del self._model
        if self._tokenizer:
            del self._tokenizer
        if self._config:
            del self._config
        self.clear_memory()