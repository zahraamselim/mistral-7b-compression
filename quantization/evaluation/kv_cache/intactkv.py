"""
IntactKV: Research-accurate implementation.

Paper: "IntactKV: Improving Large Language Model Quantization by 
        Keeping Pivot Tokens Intact" (ACL 2024 Findings)

Official repo: https://github.com/ruikangliu/IntactKV

Key insight: Some tokens (pivot tokens) receive most attention.
Preserving their KV cache in FP16 dramatically improves quantization.
"""

import torch
from typing import Tuple, Set

from .base import KVCacheQuantizer, QuantizedKVCache
from .config import IntactKVConfig


class IntactKVCache(KVCacheQuantizer):
    """
    IntactKV: Preserve pivot tokens in full precision.
    
    Can wrap any base quantization method.
    """
    
    def __init__(self, config: IntactKVConfig):
        super().__init__(config)
        self.config: IntactKVConfig = config
        
        # Track which positions are pivot tokens
        self.pivot_positions: Set[int] = set()
        
        # Initialize with fixed positions if specified
        if config.fixed_pivot_positions:
            self.pivot_positions.update(config.fixed_pivot_positions)
        
        # Base quantizer for non-pivot tokens (4-bit symmetric)
        self.qmax = 2 ** (config.key_bits - 1) - 1
        self.qmin = -2 ** (config.key_bits - 1)
    
    def _is_pivot_token(self, position: int) -> bool:
        """Check if position is a pivot token."""
        # Always preserve initial tokens (attention sinks)
        if position < self.config.preserve_initial_tokens:
            return True
        
        # Check if detected as pivot
        if position in self.pivot_positions:
            return True
        
        return False
    
    def quantize_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> QuantizedKVCache:
        """
        Quantize with pivot token preservation.
        """
        # Check if this is a pivot token
        if self._is_pivot_token(position):
            # Preserve in FP16
            return QuantizedKVCache(
                q_keys=keys.to(torch.float16),
                q_values=values.to(torch.float16),
                is_quantized=False,  # Mark as not quantized
                layer_idx=layer_idx,
                position=position
            )
        
        # Not a pivot token - quantize normally
        # Simple symmetric quantization
        k_absmax = keys.abs().amax(dim=-1, keepdim=True)
        k_scale = k_absmax / self.qmax
        k_scale = k_scale.clamp(min=1e-5)
        q_keys = (keys / k_scale).round().clamp(self.qmin, self.qmax)
        
        v_absmax = values.abs().amax(dim=-1, keepdim=True)
        v_scale = v_absmax / self.qmax
        v_scale = v_scale.clamp(min=1e-5)
        q_values = (values / v_scale).round().clamp(self.qmin, self.qmax)
        
        return QuantizedKVCache(
            q_keys=q_keys.to(torch.int8),
            q_values=q_values.to(torch.int8),
            key_scales=k_scale.to(torch.float16),
            value_scales=v_scale.to(torch.float16),
            is_quantized=True,
            layer_idx=layer_idx,
            position=position
        )
    
    def detect_pivot_tokens(
        self,
        attention_scores: torch.Tensor,
        threshold: float = None
    ) -> None:
        """
        Detect pivot tokens from attention scores.
        
        From paper: tokens that receive high attention across many queries.
        
        Args:
            attention_scores: [batch, num_heads, seq_len, seq_len]
            threshold: Attention score threshold (uses config if None)
        """
        if not self.config.detect_pivot_tokens:
            return
        
        threshold = threshold or self.config.pivot_threshold
        
        # Average attention received by each token
        attention_per_token = attention_scores.mean(dim=(0, 1, 2))  # [seq_len]
        
        # Find high-attention tokens
        pivot_mask = attention_per_token > threshold
        pivot_positions = torch.where(pivot_mask)[0].tolist()
        
        # Update pivot set (limit to max_pivot_tokens)
        self.pivot_positions.update(pivot_positions[:self.config.max_pivot_tokens])
    
    def dequantize_kv(
        self,
        quantized_cache: QuantizedKVCache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize IntactKV cache."""
        if not quantized_cache.is_quantized:
            # Was pivot token, stored in FP16
            return quantized_cache.q_keys, quantized_cache.q_values
        
        # Dequantize non-pivot tokens
        keys = quantized_cache.q_keys.float() * quantized_cache.key_scales
        values = quantized_cache.q_values.float() * quantized_cache.value_scales
        
        return keys, values