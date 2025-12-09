"""
Configuration classes for KV cache quantization methods.

Each configuration class corresponds to a specific quantization method
and contains all necessary hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class QuantizationType(Enum):
    """Types of quantization supported."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    NON_UNIFORM = "non_uniform"


@dataclass
class KVCacheConfig:
    """Base configuration for KV cache quantization."""
    
    # Quantization bits
    key_bits: int = 4
    value_bits: int = 4
    
    # Attention sink configuration
    attention_sink_size: int = 128  # Number of initial tokens to keep in FP16
    use_attention_sinks: bool = True
    
    # Calibration
    use_calibration: bool = True
    calibration_samples: int = 128
    
    # Device
    device: str = "cuda"
    dtype: str = "float16"  # Dtype for dequantized values


@dataclass
class KVQuantConfig(KVCacheConfig):
    """
    Configuration for KVQuant method.
    
    Paper: "KVQuant: Towards 10 Million Context Length LLM Inference 
            with KV Cache Quantization" (NeurIPS 2024)
    
    Key features:
    - Per-channel key quantization
    - Per-token value quantization  
    - Non-uniform quantization (NUQ)
    - Dense-and-sparse quantization for outliers
    - Pre-RoPE quantization
    - Attention sink awareness
    """
    
    # KVQuant-specific settings
    key_bits: int = 3  # Paper uses 3-bit for keys
    value_bits: int = 3  # Paper uses 3-bit for values
    
    # Per-channel vs per-token
    per_channel_keys: bool = True
    per_token_values: bool = True
    
    # Non-uniform quantization
    use_nuq: bool = True
    nuq_method: str = "kmeans"  # or "lloyd"
    num_codebook_entries: int = 8  # 2^3 for 3-bit
    
    # Dense-and-sparse quantization
    use_dense_sparse: bool = True
    outlier_threshold: float = 3.5  # Std devs for outlier detection
    outlier_fraction: float = 0.05  # Max 5% outliers
    
    # Pre-RoPE quantization (quantize before rotary embeddings)
    pre_rope_quant: bool = True
    
    # Attention sink specific
    attention_sink_size: int = 128  # Paper uses 5-128 depending on model
    preserve_sink_in_fp16: bool = True
    
    # Calibration (uses Fisher information in paper)
    use_fisher_information: bool = False  # Requires gradient computation
    fisher_samples: int = 128


@dataclass
class KIVIConfig(KVCacheConfig):
    """
    Configuration for KIVI method.
    
    Paper: "KIVI: A Tuning-Free Asymmetric 2bit Quantization 
            for KV Cache" (ICML 2024)
    
    Key features:
    - 2-bit asymmetric quantization
    - Per-channel for keys
    - Per-token for values
    - Streaming-compatible
    - No calibration required (tuning-free)
    """
    
    # KIVI uses 2-bit for aggressive compression
    key_bits: int = 2
    value_bits: int = 2
    
    # Asymmetric quantization (with zero-point)
    quantization_type: QuantizationType = QuantizationType.ASYMMETRIC
    
    # Per-channel keys, per-token values
    per_channel_keys: bool = True
    per_token_values: bool = True
    
    # Tuning-free (no calibration)
    use_calibration: bool = False
    
    # Attention sinks not used in original KIVI
    use_attention_sinks: bool = False
    attention_sink_size: int = 0
    
    # Group size for quantization
    group_size: int = 128  # Quantize in groups


@dataclass
class IntactKVConfig(KVCacheConfig):
    """
    Configuration for IntactKV method.
    
    Paper: "IntactKV: Improving Large Language Model Quantization 
            by Keeping Pivot Tokens Intact" (ACL 2024 Findings)
    
    Key features:
    - Preserves KV cache of "pivot tokens" (attention sinks) in FP16
    - Pivot tokens are not just first N tokens, but tokens with high attention
    - Can be calibrated or used directly
    - Works with any base quantization method
    """
    
    # Base quantization for non-pivot tokens
    key_bits: int = 4
    value_bits: int = 4
    
    # Pivot token detection
    detect_pivot_tokens: bool = True
    pivot_detection_method: str = "attention_score"  # or "fixed_position"
    max_pivot_tokens: int = 256  # Maximum number of pivot tokens
    pivot_threshold: float = 0.1  # Attention score threshold
    
    # Fixed position pivots (if not using detection)
    fixed_pivot_positions: Optional[List[int]] = field(default_factory=lambda: [0])  # [BOS]
    
    # Preserve initial tokens (attention sinks)
    preserve_initial_tokens: int = 32  # Always preserve first N tokens
    
    # Calibration of pivot tokens (optional)
    calibrate_pivots: bool = False  # Train pivot positions
    calibration_lr: float = 1e-4
    calibration_steps: int = 100
    
    # Base quantization for non-pivot tokens (can be any method)
    base_quantization_type: QuantizationType = QuantizationType.SYMMETRIC


@dataclass
class AQUAKVConfig(KVCacheConfig):
    """
    Configuration for AQUA-KV style asymmetric quantization.
    
    This is a standard asymmetric quantization approach with zero-point
    for better range coverage. Not from a specific paper, but a common
    technique used in KV cache quantization.
    
    Key features:
    - Asymmetric quantization with learned zero-point
    - Per-channel or per-tensor
    - Works with various bit-widths
    """
    
    # Flexible bit-width
    key_bits: int = 4
    value_bits: int = 4
    
    # Asymmetric with zero-point
    quantization_type: QuantizationType = QuantizationType.ASYMMETRIC
    
    # Granularity
    per_channel_keys: bool = True
    per_channel_values: bool = False  # Per-tensor for values
    
    # Zero-point handling
    learn_zero_point: bool = False  # Fixed vs learned
    zero_point_domain: str = "int"  # "int" or "float"
    
    # Calibration
    use_calibration: bool = True
    calibration_method: str = "minmax"  # or "percentile", "mse"
    
    # Clipping (for percentile calibration)
    clip_ratio: float = 0.99  # Clip outliers beyond 99th percentile


# Factory function for creating configs
def create_config(method: str, **kwargs) -> KVCacheConfig:
    """
    Create configuration for specified quantization method.
    
    Args:
        method: One of ['kvquant', 'kivi', 'intactkv', 'aqua']
        **kwargs: Override default configuration values
        
    Returns:
        Appropriate configuration instance
    """
    method = method.lower()
    
    if method == 'kvquant':
        return KVQuantConfig(**kwargs)
    elif method == 'kivi':
        return KIVIConfig(**kwargs)
    elif method == 'intactkv':
        return IntactKVConfig(**kwargs)
    elif method in ['aqua', 'aqua-kv', 'aquakv']:
        return AQUAKVConfig(**kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: kvquant, kivi, intactkv, aqua"
        )
