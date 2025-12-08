"""HuggingFace Transformers model implementation."""

import torch
import logging
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class HuggingFaceModel(ModelInterface):
    """
    Standard HuggingFace Transformers model.
    
    Supports any model on HuggingFace Hub or local path.
    Compatible with FP16, BF16, INT8, and NF4 quantization via bitsandbytes.
    """
    
    def load(
        self,
        model_path: str,
        quantization_config: Optional[Any] = None,
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load HuggingFace model and tokenizer.
        
        Args:
            model_path: Path or HuggingFace hub ID
            quantization_config: BitsAndBytesConfig for quantization
            torch_dtype: Model data type
            device_map: Device mapping strategy
            trust_remote_code: Whether to trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional AutoModelForCausalLM arguments
        """
        logger.info(f"Loading HuggingFace model: {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            self.model.eval()
            
            if self.model.generation_config.pad_token_id is None:
                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            
            self.device = str(next(self.model.parameters()).device)
            
            info = self.get_model_info()
            logger.info(f"Model loaded successfully")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Size: {info.get('size_gb', 0):.2f} GB")
            logger.info(f"  Parameters: {info.get('num_parameters', 0):,}")
            logger.info(f"  Device: {info['device']}")
            
            if 'gpu_memory_allocated_gb' in info:
                logger.info(f"  GPU Memory: {info['gpu_memory_allocated_gb']:.2f} GB allocated")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        return_full_text: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            return_full_text: Return prompt + generation
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        if return_full_text:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        
        return response.strip()
    
    def get_loglikelihood(self, text: str, context: str = "") -> float:
        """
        Calculate log-likelihood of text given context.
        
        Args:
            text: Full text to evaluate
            context: Context prefix
            
        Returns:
            Log probability of continuation
        """
        context_ids = self.tokenizer.encode(
            context,
            add_special_tokens=True
        ) if context else []
        
        full_ids = self.tokenizer.encode(text, add_special_tokens=True)
        continuation_start = len(context_ids)
        
        input_tensor = torch.tensor([full_ids]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs.logits
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        total_log_prob = 0.0
        for i in range(continuation_start, len(full_ids)):
            if i == 0:
                continue
            token_id = full_ids[i]
            token_log_prob = log_probs[0, i-1, token_id].item()
            total_log_prob += token_log_prob
        
        return total_log_prob
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            **kwargs: Additional forward arguments
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            return outputs.logits