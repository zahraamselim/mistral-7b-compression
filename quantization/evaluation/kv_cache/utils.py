"""Utility functions for KV cache quantization."""

import torch
from typing import Dict, Any

from .base import KVCacheQuantizer
from .kvquant import KVQuantCache
from .kivi import KIVICache
from .intactkv import IntactKVCache
from .aqua_kv import AQUAKVCache
from .config import create_config


def create_quantizer(method: str, **config_kwargs) -> KVCacheQuantizer:
    """
    Factory function to create quantizer.
    
    Args:
        method: 'kvquant', 'kivi', 'intactkv', 'aqua'
        **config_kwargs: Configuration overrides
    
    Returns:
        KVCacheQuantizer instance
    """
    config = create_config(method, **config_kwargs)
    
    method = method.lower()
    if method == 'kvquant':
        return KVQuantCache(config)
    elif method == 'kivi':
        return KIVICache(config)
    elif method == 'intactkv':
        return IntactKVCache(config)
    elif method in ['aqua', 'aqua-kv']:
        return AQUAKVCache(config)
    else:
        raise ValueError(f"Unknown method: {method}")


def estimate_kv_cache_memory(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    bits_per_param: int = 16
) -> Dict[str, float]:
    """
    Estimate KV cache memory requirements.
    
    Returns:
        Dict with memory in MB for different scenarios
    """
    # KV cache: 2 (K+V) * layers * batch * heads * seq * head_dim
    num_elements = 2 * num_layers * batch_size * num_heads * seq_len * head_dim
    
    bytes_per_element = bits_per_param / 8
    total_bytes = num_elements * bytes_per_element
    total_mb = total_bytes / (1024 ** 2)
    
    return {
        'fp16_mb': total_mb,
        'int8_mb': total_mb / 2,
        'int4_mb': total_mb / 4,
        'int3_mb': total_mb * 3 / 16,
        'int2_mb': total_mb / 8,
    }