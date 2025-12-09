"""
AQUA-KV: Standard asymmetric quantization.

This is a common baseline approach for KV cache quantization.
Not from a specific paper, but represents the "AsymKV" and "AQUA-KV"
approaches mentioned in your thesis.
"""

import torch
from typing import Tuple

from .base import KVCacheQuantizer, QuantizedKVCache
from .config import AQUAKVConfig


class AQUAKVCache(KVCacheQuantizer):
    """Standard asymmetric quantization with zero-point."""
    
    def __init__(self, config: AQUAKVConfig):
        super().__init__(config)
        self.config: AQUAKVConfig = config
        
        self.qmax = 2 ** config.key_bits - 1
        self.qmin = 0
    
    def quantize_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> QuantizedKVCache:
        """Standard asymmetric quantization."""
        # Per-channel keys
        k_min = keys.amin(dim=-1, keepdim=True)
        k_max = keys.amax(dim=-1, keepdim=True)
        k_scale = (k_max - k_min) / self.qmax
        k_scale = k_scale.clamp(min=1e-5)
        k_zero = (-k_min / k_scale).round()
        q_keys = ((keys / k_scale) + k_zero).round().clamp(self.qmin, self.qmax)
        
        # Per-tensor values (simpler)
        v_min = values.amin()
        v_max = values.amax()
        v_scale = (v_max - v_min) / self.qmax
        v_scale = v_scale.clamp(min=1e-5)
        v_zero = (-v_min / v_scale).round()
        q_values = ((values / v_scale) + v_zero).round().clamp(self.qmin, self.qmax)
        
        return QuantizedKVCache(
            q_keys=q_keys.to(torch.uint8),
            q_values=q_values.to(torch.uint8),
            key_scales=k_scale.to(torch.float16),
            value_scales=v_scale.to(torch.float16).expand_as(k_scale),
            key_zero_points=k_zero.to(torch.float16),
            value_zero_points=v_zero.to(torch.float16).expand_as(k_zero),
            is_quantized=True,
            layer_idx=layer_idx,
            position=position
        )
    
    def dequantize_kv(
        self,
        quantized_cache: QuantizedKVCache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize asymmetric."""
        keys = (
            (quantized_cache.q_keys.float() - quantized_cache.key_zero_points) 
            * quantized_cache.key_scales
        )
        
        values = (
            (quantized_cache.q_values.float() - quantized_cache.value_zero_points) 
            * quantized_cache.value_scales
        )
        
        return keys, values