"""
KIVI: Research-accurate implementation.

Paper: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (ICML 2024)

Official repo: https://github.com/jy-yuan/KIVI

Key features:
- 2-bit asymmetric quantization (most aggressive)
- Per-channel for keys, per-token for values
- No calibration needed (tuning-free)
- Streaming-compatible
"""

import torch
from typing import Tuple

from .base import KVCacheQuantizer, QuantizedKVCache
from .config import KIVIConfig


class KIVICache(KVCacheQuantizer):
    """
    KIVI 2-bit asymmetric quantization.
    
    From paper: achieves 2-bit compression with minimal accuracy loss.
    """
    
    def __init__(self, config: KIVIConfig):
        super().__init__(config)
        self.config: KIVIConfig = config
        
        # 2-bit: values 0, 1, 2, 3
        self.qmax = 3
        self.qmin = 0
    
    def quantize_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> QuantizedKVCache:
        """
        KIVI 2-bit asymmetric quantization.
        
        Formula from paper:
        q = round((x - min) / scale)
        where scale = (max - min) / (2^bits - 1)
        """
        # Quantize keys (per-channel)
        q_keys, k_scales, k_zeros = self._quantize_asymmetric_perchannel(keys)
        
        # Quantize values (per-token)
        q_values, v_scales, v_zeros = self._quantize_asymmetric_pertoken(values)
        
        return QuantizedKVCache(
            q_keys=q_keys,
            q_values=q_values,
            key_scales=k_scales,
            value_scales=v_scales,
            key_zero_points=k_zeros,
            value_zero_points=v_zeros,
            is_quantized=True,
            layer_idx=layer_idx,
            position=position
        )
    
    def _quantize_asymmetric_perchannel(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-channel asymmetric quantization for keys."""
        # Min/max per channel (last dim)
        min_val = tensor.amin(dim=-1, keepdim=True)
        max_val = tensor.amax(dim=-1, keepdim=True)
        
        # Scale and zero-point
        scale = (max_val - min_val) / self.qmax
        scale = scale.clamp(min=1e-5)
        zero_point = (-min_val / scale).round()
        
        # Quantize
        q_tensor = ((tensor / scale) + zero_point).round().clamp(self.qmin, self.qmax)
        
        return (
            q_tensor.to(torch.uint8),
            scale.to(torch.float16),
            zero_point.to(torch.float16)
        )
    
    def _quantize_asymmetric_pertoken(
        self,
        tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token asymmetric quantization for values."""
        # Min/max per token
        min_val = tensor.amin(dim=-1, keepdim=True)
        max_val = tensor.amax(dim=-1, keepdim=True)
        
        # Scale and zero-point
        scale = (max_val - min_val) / self.qmax
        scale = scale.clamp(min=1e-5)
        zero_point = (-min_val / scale).round()
        
        # Quantize
        q_tensor = ((tensor / scale) + zero_point).round().clamp(self.qmin, self.qmax)
        
        return (
            q_tensor.to(torch.uint8),
            scale.to(torch.float16),
            zero_point.to(torch.float16)
        )
    
    def dequantize_kv(
        self,
        quantized_cache: QuantizedKVCache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize KIVI 2-bit cache."""
        # Dequantize: x = (q - zero_point) * scale
        keys = (
            (quantized_cache.q_keys.float() - quantized_cache.key_zero_points) 
            * quantized_cache.key_scales
        )
        
        values = (
            (quantized_cache.q_values.float() - quantized_cache.value_zero_points) 
            * quantized_cache.value_scales
        )
        
        return keys, values