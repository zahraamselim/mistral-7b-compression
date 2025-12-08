"""ExLlamaV2 model implementation for optimized GPTQ inference."""

import torch
import logging
from typing import Optional
from types import SimpleNamespace

from core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class ExLlamaV2Model(ModelInterface):
    """
    ExLlamaV2 optimized GPTQ model interface.
    
    Requires: exllamav2
    Install: pip install exllamav2
    
    Provides faster inference for GPTQ quantized models.
    """
    
    def load(
        self,
        model_path: str,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        lazy_load: bool = True,
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load GPTQ model with ExLlamaV2.
        
        Args:
            model_path: Path to GPTQ quantized model
            max_seq_len: Maximum sequence length
            max_batch_size: Maximum batch size
            lazy_load: Use lazy loading for cache
            trust_remote_code: Whether to trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional arguments
        """
        try:
            from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
        except ImportError:
            raise ImportError(
                "exllamav2 required. Install: pip install exllamav2"
            )
        
        logger.info(f"Loading ExLlamaV2 model: {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        
        try:
            self.config = ExLlamaV2Config()
            self.config.model_dir = model_path
            self.config.prepare()
            
            self.config.max_seq_len = max_seq_len
            self.config.max_batch_size = max_batch_size
            
            self.model = ExLlamaV2(self.config)
            
            self.cache = ExLlamaV2Cache(self.model, lazy=lazy_load)
            
            logger.info("Loading model weights...")
            self.model.load_autosplit(self.cache)
            
            self.tokenizer = ExLlamaV2Tokenizer(self.config)
            
            if not hasattr(self.tokenizer, 'pad_token_id'):
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            info = self.get_model_info()
            logger.info("ExLlamaV2 model loaded successfully")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Device: {info['device']}")
            logger.info(f"  Max seq len: {max_seq_len}")
            
        except Exception as e:
            logger.error(f"Failed to load ExLlamaV2 model: {e}")
            raise
    
    def tokenize(self, text: str, add_special_tokens: bool = True, padding: bool = False, return_tensors: str = None):
        """
        Tokenize text using ExLlamaV2Tokenizer.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad (ignored for ExLlamaV2)
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        input_ids = self.tokenizer.encode(text, add_bos=add_special_tokens, add_eos=False)
        
        if return_tensors == 'pt':
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        return {'input_ids': input_ids}
    
    def generate(
        self,
        prompt: str = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        pad_token_id: int = None,
        eos_token_id: int = None,
        return_full_text: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from prompt or input_ids.
        
        Args:
            prompt: Input text string
            input_ids: Token IDs tensor
            attention_mask: Attention mask tensor
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            return_full_text: Return prompt + generation
            kwargs: Additional arguments
            
        Returns:
            Generated text string
        """
        try:
            from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler
        except ImportError:
            raise ImportError("exllamav2 required")
        
        if prompt is None and input_ids is None:
            raise ValueError("Either prompt or input_ids required")
        
        if input_ids is not None:
            if input_ids.dim() == 2:
                input_ids = input_ids[0]
            
            if return_full_text:
                prompt_text = self.tokenizer.decode(input_ids)
            else:
                prompt_text = ""
        else:
            prompt_text = prompt if return_full_text else ""
            input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        
        generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature if do_sample and temperature > 0 else 0.0
        settings.top_p = top_p
        settings.top_k = top_k
        settings.token_repetition_penalty = 1.15
        
        generator.begin_stream(input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids, settings)
        
        output_text = ""
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            chunk, eos, _ = generator.stream()
            output_text += chunk
            tokens_generated += 1
            
            if eos:
                break
        
        result = prompt_text + output_text if return_full_text else output_text
        
        # Store the actual token count for retrieval
        if hasattr(self, '_last_generated_tokens'):
            self._last_generated_tokens = tokens_generated
        else:
            self._last_generated_tokens = tokens_generated
            
        return result
    
    def get_last_generated_token_count(self):
        """Get the number of tokens generated in the last generate() call."""
        return getattr(self, '_last_generated_tokens', 0)
    
    def get_loglikelihood(self, text: str, context: str = "") -> float:
        """Calculate log-likelihood."""
        full_text = context + text if context else text
        input_ids = self.tokenizer.encode(full_text)
        
        context_len = len(self.tokenizer.encode(context)) if context else 0
        
        self.cache.current_seq_len = 0
        
        logits = self.model.forward(input_ids, self.cache, input_mask=None)
        
        log_probs = torch.log_softmax(logits, dim=-1)
        
        total_log_prob = 0.0
        for i in range(context_len, input_ids.shape[1] - 1):
            token_id = input_ids[0, i + 1].item()
            token_log_prob = log_probs[0, i, token_id].item()
            total_log_prob += token_log_prob
        
        return total_log_prob
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        """
        Forward pass with optional labels for loss calculation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len] or [seq_len]
            labels: Optional labels for loss calculation
            **kwargs: Additional arguments
            
        Returns:
            Namespace with logits and optionally loss
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        self.cache.current_seq_len = 0
        logits = self.model.forward(input_ids, self.cache, input_mask=None)
        
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            return SimpleNamespace(logits=logits, loss=loss)
        
        return SimpleNamespace(logits=logits)
    
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """Make model callable for forward pass compatibility."""
        return self.forward(input_ids, **kwargs)
    
    def parameters(self):
        """Dummy parameters method for compatibility."""
        return iter([])
    
    def eval(self):
        """Set model to eval mode (no-op for ExLlamaV2)."""
        return self
    
    def get_model_info(self):
        """Get model information."""
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
        }
        
        if hasattr(self.config, 'max_seq_len'):
            info["max_seq_len"] = self.config.max_seq_len
        
        estimated_params = 7_240_000_000
        info["num_parameters"] = estimated_params
        
        bits_per_param = 4.0
        if self.model_path and '2bit' in str(self.model_path).lower():
            bits_per_param = 2.0
        elif self.model_path and '3bit' in str(self.model_path).lower():
            bits_per_param = 3.0
        
        info["size_gb"] = (estimated_params * bits_per_param / 8) / (1024**3)
        
        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
        
        return info
    
    def get_lm_eval_model(self):
        """Get lm-eval compatible model wrapper."""
        logger.warning("ExLlamaV2 does not natively support lm-eval")
        logger.info("Creating HuggingFace wrapper for task evaluation")
        
        try:
            from transformers import AutoTokenizer, AutoConfig
            from auto_gptq import AutoGPTQForCausalLM
            from lm_eval.models.huggingface import HFLM
            
            hf_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=False)
            
            hf_model = AutoGPTQForCausalLM.from_quantized(
                self.model_path,
                device_map="auto",
                use_safetensors=True,
                trust_remote_code=False,
                use_triton=False
            )
            
            return HFLM(
                pretrained=hf_model,
                tokenizer=hf_tokenizer,
                batch_size=1
            )
        except ImportError as e:
            logger.error(f"Missing dependencies for lm-eval: {e}")
            logger.error("Install: pip install optimum auto-gptq")
            return None
        except Exception as e:
            logger.error(f"Could not create lm-eval wrapper: {e}")
            return None