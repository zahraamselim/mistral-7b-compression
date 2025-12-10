"""
HuggingFace Model Implementation

Implements the ModelInterface for HuggingFace transformer models.
"""

import gc
import time
from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_interface import ModelInterface, GenerationConfig, ModelOutput


class HuggingFaceModel(ModelInterface):
    """HuggingFace transformer model implementation"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path, device)
    
    def load(self):
        """Load model and tokenizer"""
        print(f"Loading model: {self.model_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded on: {self._model.device}")
    
    def generate(self, prompt: str, config: GenerationConfig, return_attentions: bool = False) -> ModelOutput:
        """Generate text from prompt"""
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self._model.generate(
                inputs.input_ids,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p if config.do_sample else 1.0,
                top_k=config.top_k if config.do_sample else 50,
                pad_token_id=self._tokenizer.eos_token_id,
                output_attentions=return_attentions,
                return_dict_in_generate=return_attentions
            )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if return_attentions:
            generated_ids = outputs.sequences
            attentions = outputs.attentions if hasattr(outputs, 'attentions') else None
        else:
            generated_ids = outputs
            attentions = None
        
        generated_text = self._tokenizer.decode(
            generated_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        num_tokens = generated_ids.shape[1] - inputs.input_ids.shape[1]
        
        return ModelOutput(
            generated_ids=generated_ids,
            generated_text=generated_text,
            attentions=attentions,
            num_generated_tokens=num_tokens,
            latency_ms=latency_ms
        )
    
    def get_perplexity(self, text: str, max_length: int = 512) -> float:
        """Calculate perplexity on text"""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self._model.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        return torch.exp(loss).item()
    
    def encode(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Tokenize text"""
        return self._tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True if max_length else False
        ).to(self._model.device)
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs"""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info"""
        config = self._model.config
        
        # Get dtype from parameters, not model object
        try:
            first_param = next(self._model.parameters())
            dtype = str(first_param.dtype)
        except (StopIteration, AttributeError):
            dtype = "unknown"
        
        return {
            "model_path": self.model_path,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "dtype": dtype
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
        del self._model
        del self._tokenizer
        self.clear_memory()