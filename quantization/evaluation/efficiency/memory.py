"""
Memory measurement utilities.

Measures model size, peak memory usage, and memory efficiency.
"""

import logging
from typing import Dict, Tuple, Optional

import torch

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


def get_model_size(model: ModelInterface) -> Tuple[float, int]:
    """
    Calculate model size including parameters and buffers.
    
    Args:
        model: ModelInterface instance
        
    Returns:
        Tuple of (size_gb, size_bytes)
    """
    try:
        param_size = sum(
            p.element_size() * p.numel() 
            for p in model.model.parameters()
        )
        buffer_size = sum(
            b.element_size() * b.numel() 
            for b in model.model.buffers()
        )
        
        total_bytes = param_size + buffer_size
        size_gb = total_bytes / (1024**3)
        
        logger.info(f"Model size: {size_gb:.3f} GB ({total_bytes:,} bytes)")
        logger.debug(f"  Parameters: {param_size / (1024**3):.3f} GB")
        logger.debug(f"  Buffers: {buffer_size / (1024**3):.3f} GB")
        
        return size_gb, total_bytes
        
    except Exception as e:
        logger.error(f"Failed to calculate model size: {e}")
        return 0.0, 0


def get_parameter_count(model: ModelInterface) -> Dict[str, int]:
    """
    Get detailed parameter counts.
    
    Args:
        model: ModelInterface instance
        
    Returns:
        Dictionary with parameter counts
    """
    try:
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        non_trainable_params = total_params - trainable_params
        
        logger.info(
            f"Parameters: {total_params:,} total "
            f"({trainable_params:,} trainable)"
        )
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
        
    except Exception as e:
        logger.error(f"Failed to count parameters: {e}")
        return {'total': 0, 'trainable': 0, 'non_trainable': 0}


def get_bits_per_param(model: ModelInterface) -> float:
    """
    Calculate bits per parameter (handles quantization).
    
    Args:
        model: ModelInterface instance
        
    Returns:
        Average bits per parameter
    """
    try:
        sample_param = next(model.model.parameters())
        
        # Check for quantization
        if hasattr(sample_param, 'quant_state'):
            quant_state = sample_param.quant_state
            quant_type = str(getattr(quant_state, 'quant_type', 'unknown')).lower()
            
            if 'nf4' in quant_type or 'int4' in quant_type or '4bit' in quant_type:
                theoretical_bits = 4.0
            elif 'int8' in quant_type or 'fp8' in quant_type or '8bit' in quant_type:
                theoretical_bits = 8.0
            else:
                theoretical_bits = 16.0
            
            # Calculate actual bits from model size
            total_params = sum(p.numel() for p in model.model.parameters())
            _, size_bytes = get_model_size(model)
            actual_bits = (size_bytes * 8) / total_params if total_params > 0 else theoretical_bits
            
            # Use theoretical if close to actual
            if abs(actual_bits - theoretical_bits) < 2.0:
                bits = theoretical_bits
            else:
                bits = actual_bits
            
            logger.info(f"Quantized model: {bits:.1f} bits/param (type: {quant_type})")
            return bits
        
        # Check data type
        dtype = sample_param.dtype
        dtype_bits = {
            torch.float32: 32.0,
            torch.float16: 16.0,
            torch.bfloat16: 16.0,
            torch.float64: 64.0,
            torch.int8: 8.0,
            torch.int32: 32.0,
            torch.int64: 64.0
        }
        bits = dtype_bits.get(dtype, 16.0)
        
        logger.info(f"Model dtype: {dtype}, bits/param: {bits}")
        return bits
        
    except Exception as e:
        logger.warning(f"Could not determine bits per param: {e}, using default 16.0")
        return 16.0


def get_peak_memory(use_cuda: bool, device_id: int = 0) -> float:
    """
    Get peak memory usage in MB.
    
    Args:
        use_cuda: Whether using CUDA
        device_id: CUDA device ID
        
    Returns:
        Peak memory in MB
    """
    if not use_cuda or not torch.cuda.is_available():
        return 0.0
    
    try:
        peak_mb = torch.cuda.max_memory_allocated(device_id) / (1024**2)
        logger.info(f"Peak GPU memory: {peak_mb:.2f} MB")
        return peak_mb
        
    except Exception as e:
        logger.warning(f"Failed to get peak memory: {e}")
        return 0.0


def get_current_memory(use_cuda: bool, device_id: int = 0) -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Args:
        use_cuda: Whether using CUDA
        device_id: CUDA device ID
        
    Returns:
        Dictionary with memory statistics in MB
    """
    if not use_cuda or not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'free_mb': 0.0,
            'total_mb': 0.0
        }
    
    try:
        allocated = torch.cuda.memory_allocated(device_id) / (1024**2)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**2)
        
        props = torch.cuda.get_device_properties(device_id)
        total = props.total_memory / (1024**2)
        free = total - allocated
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total
        }
        
    except Exception as e:
        logger.warning(f"Failed to get current memory: {e}")
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'free_mb': 0.0,
            'total_mb': 0.0
        }


def get_memory_efficiency(model_size_gb: float, peak_memory_mb: float) -> Optional[float]:
    """
    Calculate memory efficiency (model size / peak memory).
    
    Args:
        model_size_gb: Model size in GB
        peak_memory_mb: Peak memory in MB
        
    Returns:
        Memory efficiency ratio (ideally close to 1.0)
    """
    if peak_memory_mb == 0:
        return None
    
    peak_memory_gb = peak_memory_mb / 1024
    efficiency = model_size_gb / peak_memory_gb if peak_memory_gb > 0 else 0.0
    
    logger.info(f"Memory efficiency: {efficiency:.2f} (model/peak ratio)")
    return efficiency


def reset_memory_stats(use_cuda: bool, device_id: int = 0) -> None:
    """
    Reset peak memory statistics.
    
    Args:
        use_cuda: Whether using CUDA
        device_id: CUDA device ID
    """
    if use_cuda and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device_id)
            torch.cuda.empty_cache()
            logger.debug("Reset CUDA memory statistics")
        except Exception as e:
            logger.warning(f"Failed to reset memory stats: {e}")


def estimate_kv_cache_size(
    model: ModelInterface,
    batch_size: int = 1,
    sequence_length: int = 2048
) -> Optional[float]:
    """
    Estimate KV cache size for transformer models.
    
    Args:
        model: ModelInterface instance
        batch_size: Batch size
        sequence_length: Sequence length
        
    Returns:
        Estimated KV cache size in MB, or None if estimation fails
    """
    try:
        config = model.model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        
        sample_param = next(model.model.parameters())
        dtype_bytes = sample_param.element_size()
        
        # KV cache: 2 (K and V) * num_layers * batch * num_heads * seq_len * head_dim * bytes
        kv_cache_bytes = (
            2 * num_layers * batch_size * num_heads * sequence_length * head_dim * dtype_bytes
        )
        kv_cache_mb = kv_cache_bytes / (1024**2)
        
        logger.info(
            f"Estimated KV cache size: {kv_cache_mb:.2f} MB "
            f"(batch={batch_size}, seq_len={sequence_length})"
        )
        
        return kv_cache_mb
        
    except Exception as e:
        logger.warning(f"Could not estimate KV cache size: {e}")
        return None
