"""
evaluation/kv_cache/ - Research-Accurate KV Cache Quantization Implementations

Structure:
kv_cache/
├── __init__.py                  # Exports all quantizers
├── base.py                      # Abstract base class
├── kvquant.py                   # KVQuant (NeurIPS 2024) - Full implementation
├── kivi.py                      # KIVI (ICML 2024) - 2-bit asymmetric
├── intactkv.py                  # IntactKV (ACL 2024) - Pivot token preservation
├── aqua_kv.py                   # AQUA-KV style asymmetric quantization
├── config.py                    # Configuration classes
├── utils.py                     # Shared utilities
└── README.md                    # Documentation

Each file contains:
1. Research-accurate implementation following the paper
2. Citations and paper references
3. Configurable hyperparameters
4. Memory estimation utilities
"""

# evaluation/kv_cache/__init__.py
"""
KV Cache Quantization Methods for LLM Evaluation

Implements state-of-the-art KV cache quantization methods from recent research:
- KVQuant (NeurIPS 2024): Per-channel + NUQ + Dense-Sparse + Attention sinks
- KIVI (ICML 2024): 2-bit asymmetric quantization
- IntactKV (ACL 2024): Pivot token preservation
- AQUA-KV: Asymmetric quantization with zero-point
"""

from .base import KVCacheQuantizer
from .kvquant import KVQuantCache
from .kivi import KIVICache
from .intactkv import IntactKVCache
from .aqua_kv import AQUAKVCache
from .config import KVCacheConfig
from .utils import estimate_kv_cache_memory, create_quantizer

__all__ = [
    'KVCacheQuantizer',
    'KVQuantCache',
    'KIVICache',
    'IntactKVCache',
    'AQUAKVCache',
    'KVCacheConfig',
    'estimate_kv_cache_memory',
    'create_quantizer',
]

__version__ = '1.0.0'