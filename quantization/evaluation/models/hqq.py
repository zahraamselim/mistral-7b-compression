"""HQQ quantized model implementation."""

import torch
import logging
from typing import Optional
from pathlib import Path

from core.model_interface import ModelInterface

logger = logging.getLogger(__name__)


class HQQModel(ModelInterface):
    """
    HQQ (Half-Quadratic Quantization) quantized model.
    
    Requires: hqq
    Install: pip install hqq
    
    Supports 2/3/4/8-bit quantization.
    Can quantize models on-the-fly or load pre-quantized.
    """
    
    def load(
        self,
        model_path: str,
        nbits: int = 4,
        group_size: int = 64,
        axis: int = 1,
        device: str = 'cuda',
        compute_dtype: torch.dtype = torch.float16,
        save_dir: Optional[str] = None,
        force_quantize: bool = False,
        trust_remote_code: bool = False,
        model_type: str = "instruct",
        **kwargs
    ):
        """
        Load or quantize model with HQQ.
        
        Args:
            model_path: Path or HuggingFace hub ID
            nbits: Quantization bits (2, 3, 4, or 8)
            group_size: Quantization group size
            axis: Quantization axis (0 or 1)
            device: Device to load on
            compute_dtype: Compute dtype
            save_dir: Directory to save/load quantized model
            force_quantize: Force re-quantization
            trust_remote_code: Whether to trust remote code
            model_type: 'base' or 'instruct'
            **kwargs: Additional arguments
        """
        try:
            from hqq.models.hf.base import AutoHQQHFModel
            from hqq.core.quantize import BaseQuantizeConfig
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "hqq required for HQQ models. "
                "Install: pip install hqq"
            )
        
        logger.info(f"Loading/Quantizing HQQ model: {model_path}")
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if save_dir and not force_quantize:
                save_path = Path(save_dir)
                if save_path.exists() and (save_path / "qmodel.pt").exists():
                    logger.info(f"Loading pre-quantized HQQ model from {save_dir}")
                    self.model = AutoHQQHFModel.from_quantized(str(save_path))
                    self.model = self.model.to(device).eval()
                    logger.info("Pre-quantized model loaded")
                else:
                    logger.info("No pre-quantized model found, quantizing...")
                    force_quantize = True
            else:
                force_quantize = True
            
            if force_quantize:
                logger.info(f"Quantizing model with HQQ ({nbits}-bit, group_size={group_size})")
                
                quant_config = BaseQuantizeConfig(
                    nbits=nbits,
                    group_size=group_size,
                    quant_zero=True,
                    quant_scale=False,
                    axis=axis
                )
                
                self.model = AutoHQQHFModel.from_pretrained(
                    model_path,
                    torch_dtype=compute_dtype,
                    trust_remote_code=trust_remote_code
                )
                
                self.model.quantize_model(
                    quant_config=quant_config,
                    compute_dtype=compute_dtype,
                    device=device
                )
                
                self.model.eval()
                
                if save_dir:
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Saving quantized model to {save_dir}")
                    self.model.save_quantized(str(save_path))
                    logger.info("Quantized model saved")
            
            info = self.get_model_info()
            logger.info("HQQ model ready")
            logger.info(f"  Type: {info.get('model_type', 'unknown')}")
            logger.info(f"  Size: {info.get('size_gb', 0):.2f} GB")
            logger.info(f"  Parameters: {info.get('num_parameters', 0):,}")
            logger.info(f"  Device: {info['device']}")
            
            if 'gpu_memory_allocated_gb' in info:
                logger.info(f"  GPU Memory: {info['gpu_memory_allocated_gb']:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load/quantize HQQ model: {e}")
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
        """Generate text from prompt."""
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
        """Calculate log-likelihood."""
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
        """Forward pass."""
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            return outputs.logits