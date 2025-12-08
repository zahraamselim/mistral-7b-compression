"""Memory measurement utilities."""

import logging
from typing import Tuple, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def get_model_size(model) -> Tuple[float, int]:
    """
    Calculate model size including parameters and buffers.
    
    Args:
        model: Language model
        
    Returns:
        Tuple of (size_gb, size_bytes)
    """
    try:
        # Check if model has parameters method and it returns items
        if hasattr(model, 'parameters'):
            params = list(model.parameters())
            if params:
                param_size = sum(p.element_size() * p.numel() for p in params)
                buffer_size = sum(b.element_size() * b.numel() for b in model.buffers()) if hasattr(model, 'buffers') else 0
                total_bytes = param_size + buffer_size
                size_gb = total_bytes / (1024**3)
                
                logger.info(f"Model size: {size_gb:.3f} GB ({total_bytes:,} bytes)")
                logger.debug(f"  Parameters: {param_size / (1024**3):.3f} GB")
                logger.debug(f"  Buffers: {buffer_size / (1024**3):.3f} GB")
                
                return size_gb, total_bytes
        
        # For ExLlamaV2 or models without standard parameters
        # Estimate based on current GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"Model size (estimated from GPU memory): {allocated:.3f} GB")
            return allocated, int(allocated * (1024**3))
        
        logger.warning("Could not determine model size")
        return 0.0, 0
        
    except Exception as e:
        logger.error(f"Failed to calculate model size: {e}")
        return 0.0, 0


def get_parameter_count(model) -> Dict[str, int]:
    """
    Get detailed parameter counts.
    
    Args:
        model: Language model
        
    Returns:
        Dictionary with parameter counts
    """
    try:
        if hasattr(model, 'parameters'):
            params = list(model.parameters())
            if params:
                total_params = sum(p.numel() for p in params)
                trainable_params = sum(p.numel() for p in params if p.requires_grad)
                non_trainable_params = total_params - trainable_params
                
                logger.info(f"Parameters: {total_params:,} total ({trainable_params:,} trainable)")
                
                return {
                    'total': total_params,
                    'trainable': trainable_params,
                    'non_trainable': non_trainable_params
                }
        
        # For models without standard parameters (ExLlamaV2)
        # Estimate based on model info if available
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            if 'num_parameters' in info:
                total = info['num_parameters']
                logger.info(f"Parameters (estimated): {total:,} total")
                return {
                    'total': total,
                    'trainable': 0,
                    'non_trainable': total
                }
        
        logger.warning("Could not determine parameter count")
        return {'total': 0, 'trainable': 0, 'non_trainable': 0}
        
    except Exception as e:
        logger.error(f"Failed to count parameters: {e}")
        return {'total': 0, 'trainable': 0, 'non_trainable': 0}


def get_bits_per_param(model) -> float:
    """
    Calculate bits per parameter.
    
    Handles quantization detection.
    
    Args:
        model: Language model
        
    Returns:
        Average bits per parameter
    """
    try:
        # Check for ExLlamaV2 or custom quantized models
        if hasattr(model, 'config'):
            # Try to infer from model path or config
            if hasattr(model, 'model_path') and model.model_path:
                model_path_lower = str(model.model_path).lower()
                if '2bit' in model_path_lower or '2-bit' in model_path_lower:
                    logger.info("Detected 2-bit quantization from model path")
                    return 2.0
                elif '3bit' in model_path_lower or '3-bit' in model_path_lower:
                    logger.info("Detected 3-bit quantization from model path")
                    return 3.0
                elif '4bit' in model_path_lower or '4-bit' in model_path_lower or 'gptq' in model_path_lower:
                    logger.info("Detected 4-bit quantization from model path")
                    return 4.0
        
        # Try standard parameter inspection
        if hasattr(model, 'parameters'):
            params = list(model.parameters())
            if params:
                sample_param = params[0]
                
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
                    
                    total_params = sum(p.numel() for p in params)
                    _, size_bytes = get_model_size(model)
                    actual_bits = (size_bytes * 8) / total_params if total_params > 0 else theoretical_bits
                    
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
        
        # Default fallback
        logger.warning(f"Could not determine bits per param, using default 4.0 (GPTQ)")
        return 4.0
        
    except Exception as e:
        logger.warning(f"Could not determine bits per param: {e}, using default 4.0")
        return 4.0


def get_peak_memory(is_cuda: bool, device_id: int = 0) -> float:
    """
    Get peak memory usage in MB.
    
    Args:
        is_cuda: Whether using CUDA
        device_id: CUDA device ID
        
    Returns:
        Peak memory in MB
    """
    if not is_cuda or not torch.cuda.is_available():
        return 0.0
    
    try:
        peak_mb = torch.cuda.max_memory_allocated(device_id) / (1024**2)
        logger.info(f"Peak GPU memory: {peak_mb:.2f} MB")
        return peak_mb
    except Exception as e:
        logger.warning(f"Failed to get peak memory: {e}")
        return 0.0


def get_current_memory(is_cuda: bool, device_id: int = 0) -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Args:
        is_cuda: Whether using CUDA
        device_id: CUDA device ID
        
    Returns:
        Dictionary with memory statistics in MB
    """
    if not is_cuda or not torch.cuda.is_available():
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
    Calculate memory efficiency.
    
    Args:
        model_size_gb: Model size in GB
        peak_memory_mb: Peak memory in MB
        
    Returns:
        Memory efficiency ratio (ideally close to 1.0)
    """
    if peak_memory_mb == 0 or model_size_gb == 0:
        return None
    
    peak_memory_gb = peak_memory_mb / 1024
    efficiency = model_size_gb / peak_memory_gb if peak_memory_gb > 0 else 0.0
    
    logger.info(f"Memory efficiency: {efficiency:.2f} (model/peak ratio)")
    return efficiency


def reset_memory_stats(is_cuda: bool, device_id: int = 0):
    """
    Reset peak memory statistics.
    
    Args:
        is_cuda: Whether using CUDA
        device_id: CUDA device ID
    """
    if is_cuda and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device_id)
            torch.cuda.empty_cache()
            logger.debug("Reset CUDA memory statistics")
        except Exception as e:
            logger.warning(f"Failed to reset memory stats: {e}")


def estimate_kv_cache_size(
    model,
    batch_size: int = 1,
    sequence_length: int = 2048
) -> Optional[float]:
    """
    Estimate KV cache size for transformer models.
    
    Args:
        model: Language model
        batch_size: Batch size
        sequence_length: Sequence length
        
    Returns:
        Estimated KV cache size in MB
    """
    try:
        config = model.config if hasattr(model, 'config') else None
        if config is None:
            logger.warning("Model has no config, cannot estimate KV cache")
            return None
        
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layers', None))
        num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_heads', None))
        hidden_size = getattr(config, 'hidden_size', getattr(config, 'd_model', None))
        
        if num_layers is None or num_heads is None or hidden_size is None:
            logger.warning("Could not extract required config values for KV cache estimation")
            return None
        
        head_dim = hidden_size // num_heads
        
        # Estimate dtype size (assume FP16 for cache)
        dtype_bytes = 2
        
        kv_cache_bytes = (
            2 * num_layers * batch_size * num_heads * sequence_length * head_dim * dtype_bytes
        )
        kv_cache_mb = kv_cache_bytes / (1024**2)
        
        logger.info(f"Estimated KV cache size: {kv_cache_mb:.2f} MB "
                   f"(batch={batch_size}, seq_len={sequence_length})")
        
        return kv_cache_mb
        
    except Exception as e:
        logger.warning(f"Could not estimate KV cache size: {e}")
        return None