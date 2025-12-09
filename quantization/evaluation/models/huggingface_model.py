"""
HuggingFace Model Implementation

Implements the ModelInterface for HuggingFace transformers models.
"""

import time
import gc
import torch
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_interface import ModelInterface, GenerationConfig, ModelOutput
from utils import get_model_size_mb


class HuggingFaceModel(ModelInterface):
    """
    HuggingFace transformers model implementation.
    
    Supports:
    - Full precision models
    - GPTQ quantized models
    - AWQ quantized models
    - GGUF models (via transformers)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize HuggingFace model.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device placement ("auto", "cuda", "cpu")
            torch_dtype: Model dtype
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Load in 8-bit (bitsandbytes)
            load_in_4bit: Load in 4-bit (bitsandbytes)
        """
        super().__init__(model_path, device)
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._model_size_mb = None
    
    def load(self) -> None:
        """Load model and tokenizer into memory."""
        print(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        # Set pad token if not present
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit
        )
        
        self._model.eval()
        
        # Calculate model size
        self._model_size_mb = get_model_size_mb(self.model_path)
        
        print(f"✓ Model loaded")
        print(f"  Device: {self._model.device}")
        print(f"  Dtype: {self._model.dtype}")
        print(f"  Size: {self._model_size_mb:.2f} MB")
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        return_attentions: bool = False
    ) -> ModelOutput:
        """Generate text from prompt."""
        inputs = self.encode(prompt)
        input_length = inputs['input_ids'].shape[1]
        
        # Prepare generation kwargs
        gen_kwargs = {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'top_p': config.top_p,
            'top_k': config.top_k,
            'do_sample': config.do_sample,
            'num_return_sequences': config.num_return_sequences,
            'pad_token_id': config.pad_token_id or self._tokenizer.pad_token_id,
            'eos_token_id': config.eos_token_id or self._tokenizer.eos_token_id,
        }
        
        if return_attentions:
            gen_kwargs['output_attentions'] = True
            gen_kwargs['return_dict_in_generate'] = True
        
        # Time the generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract results
        if return_attentions and hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        else:
            generated_ids = outputs
            attentions = None
        
        # Decode only the generated portion
        generated_text = self._tokenizer.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True
        )
        
        num_generated = generated_ids.shape[1] - input_length
        
        return ModelOutput(
            generated_ids=generated_ids,
            generated_text=generated_text,
            attentions=attentions,
            num_generated_tokens=num_generated,
            latency_ms=latency_ms
        )
    
    def get_perplexity(self, text: str, max_length: int = 512) -> float:
        """Calculate perplexity on text."""
        inputs = self.encode(text, max_length=max_length)
        
        if inputs['input_ids'].shape[1] < 2:
            return float('inf')
        
        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
        
        return loss.item()
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text and return input tensors."""
        encoding_kwargs = {
            'return_tensors': 'pt',
            'padding': True,
            'truncation': True if max_length else False
        }
        
        if max_length:
            encoding_kwargs['max_length'] = max_length
        
        inputs = self._tokenizer(text, **encoding_kwargs)
        
        # Move to model device
        return {k: v.to(self._model.device) for k, v in inputs.items()}
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and configuration."""
        config = self._model.config.to_dict() if hasattr(self._model, 'config') else {}
        
        return {
            "model_path": self.model_path,
            "model_type": config.get("model_type", "unknown"),
            "vocab_size": config.get("vocab_size", 0),
            "hidden_size": config.get("hidden_size", 0),
            "num_layers": config.get("num_hidden_layers", 0),
            "num_attention_heads": config.get("num_attention_heads", 0),
            "dtype": str(self._model.dtype),
            "device": str(self._model.device),
            "model_size_mb": self._model_size_mb,
            "quantization": {
                "load_in_8bit": self.load_in_8bit,
                "load_in_4bit": self.load_in_4bit
            }
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {
            "model_size_mb": self._model_size_mb or 0.0,
        }
        
        if torch.cuda.is_available():
            # Reset peak stats for accurate measurement
            torch.cuda.reset_peak_memory_stats()
            
            # Run a small forward pass
            dummy_input = self.encode("test")
            with torch.no_grad():
                _ = self._model(**dummy_input)
            
            torch.cuda.synchronize()
            
            memory_stats.update({
                "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                "current_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "current_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2)
            })
        
        return memory_stats
    
    def clear_memory(self) -> None:
        """Clear GPU/CPU cache and force garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        self.clear_memory()
        print("✓ Model unloaded")
