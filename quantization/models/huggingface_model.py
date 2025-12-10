"""
HuggingFace Model Implementation

Implements the ModelInterface for HuggingFace transformer models.
Focus: Research-grade attention extraction for RAG evaluation.
"""

import gc
import time
from typing import Dict, Any, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.model_interface import ModelInterface, GenerationConfig, ModelOutput


class HuggingFaceModel(ModelInterface):
    """HuggingFace transformer model implementation with attention extraction"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        super().__init__(model_path, device)
        self._model_type = None
    
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
        
        # Detect model type from model name/config
        model_name_lower = self.model_path.lower()
        if "instruct" in model_name_lower or "chat" in model_name_lower:
            self._model_type = "instruct"
        else:
            self._model_type = "base"
        
        print(f"Model loaded on: {self._model.device}")
        print(f"Model type detected: {self._model_type}")
    
    @property
    def model_type(self):
        """Get model type (base or instruct)"""
        return self._model_type
    
    def generate(self, prompt: str, config: GenerationConfig, return_attentions: bool = False) -> ModelOutput:
        """
        Generate text from prompt.
        
        When return_attentions=True, uses custom generation loop to capture
        attention weights at each decoding step. This is necessary because
        HuggingFace's generate() method doesn't reliably return attentions.
        
        Research note: Attention extraction adds ~2-3x overhead vs standard generation.
        For large-scale experiments, consider caching or using smaller sample sizes.
        """
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_length = inputs.input_ids.shape[1]
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if return_attentions:
                # Custom generation loop for attention extraction
                # Note: This is slower but necessary for research-grade attention analysis
                generated_ids = inputs.input_ids.clone()
                all_attentions = []
                
                for step in range(config.max_new_tokens):
                    outputs = self._model(
                        input_ids=generated_ids,
                        attention_mask=torch.ones_like(generated_ids),
                        output_attentions=True
                    )
                    
                    # Capture attention weights from all layers
                    # Format: tuple of (batch, heads, seq_len, seq_len) per layer
                    all_attentions.append(outputs.attentions)
                    
                    # Generate next token
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    if config.do_sample:
                        # Sampling with temperature/top-k/top-p
                        next_token_logits = next_token_logits / config.temperature
                        
                        if config.top_k > 0:
                            indices_to_remove = next_token_logits < torch.topk(next_token_logits, config.top_k)[0][..., -1, None]
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        if config.top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > config.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Early stopping on EOS
                    if next_token.item() == self._tokenizer.eos_token_id:
                        break
                
                attentions = all_attentions
                
            else:
                # Fast generation without attention extraction
                outputs = self._model.generate(
                    inputs.input_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature if config.do_sample else 1.0,
                    top_p=config.top_p if config.do_sample else 1.0,
                    top_k=config.top_k if config.do_sample else 50,
                    pad_token_id=self._tokenizer.eos_token_id
                )
                generated_ids = outputs
                attentions = None
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        generated_text = self._tokenizer.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True
        )
        
        num_tokens = generated_ids.shape[1] - input_length
        
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
        
        try:
            first_param = next(self._model.parameters())
            dtype = str(first_param.dtype)
        except (StopIteration, AttributeError):
            dtype = "unknown"
        
        return {
            "model_path": self.model_path,
            "model_type": self._model_type,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "dtype": dtype,
            "device": str(self._model.device)
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