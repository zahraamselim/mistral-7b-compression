"""
Space Performance Evaluation

Measures all memory and storage-related metrics: model size, peak memory,
compression ratio, memory efficiency, KV cache size.
All measurement logic is contained within this file.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

import torch

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


def get_model_size(model: ModelInterface) -> Tuple[float, int]:
    """Calculate model size including parameters and buffers."""
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
    """Get detailed parameter counts."""
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
    """Calculate bits per parameter (handles quantization)."""
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
    """Get peak memory usage in MB."""
    if not use_cuda or not torch.cuda.is_available():
        return 0.0
    
    try:
        peak_mb = torch.cuda.max_memory_allocated(device_id) / (1024**2)
        logger.info(f"Peak GPU memory: {peak_mb:.2f} MB")
        return peak_mb
        
    except Exception as e:
        logger.warning(f"Failed to get peak memory: {e}")
        return 0.0


def get_memory_efficiency(model_size_gb: float, peak_memory_mb: float) -> Optional[float]:
    """Calculate memory efficiency (model size / peak memory)."""
    if peak_memory_mb == 0:
        return None
    
    peak_memory_gb = peak_memory_mb / 1024
    efficiency = model_size_gb / peak_memory_gb if peak_memory_gb > 0 else 0.0
    
    logger.info(f"Memory efficiency: {efficiency:.2f} (model/peak ratio)")
    return efficiency


def reset_memory_stats(use_cuda: bool, device_id: int = 0) -> None:
    """Reset peak memory statistics."""
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
    """Estimate KV cache size for transformer models."""
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


class SpacePerformanceEvaluator:
    """
    Evaluates all memory and storage-related metrics.
    
    Metrics measured:
    - Model size (GB): Total model footprint
    - Parameter count: Total/trainable/non-trainable parameters
    - Bits per parameter: Quantization level
    - Peak memory (MB): Maximum GPU memory during inference
    - Memory efficiency: Model size / peak memory ratio
    - KV cache size (MB): Attention cache memory requirements
    """
    
    def __init__(
        self,
        model: ModelInterface,
        use_cuda: bool = True,
        verbose: bool = False
    ):
        self.model = model
        self.use_cuda = use_cuda
        self.verbose = verbose
    
    def run(
        self,
        batch_size: int = 1,
        sequence_length: int = 2048,
        reset_stats: bool = True
    ) -> Dict[str, Any]:
        """Run all space performance benchmarks."""
        logger.info("Running space performance evaluation...")
        
        # Reset memory statistics
        if reset_stats:
            reset_memory_stats(self.use_cuda)
        
        results = {}
        
        logger.info("\n[1/6] Measuring Model Size...")
        size_gb, size_bytes = get_model_size(self.model)
        results['model_size_gb'] = size_gb
        results['model_size_bytes'] = size_bytes
        
        logger.info("\n[2/6] Counting Parameters...")
        param_counts = get_parameter_count(self.model)
        results['total_params'] = param_counts['total']
        results['trainable_params'] = param_counts['trainable']
        results['non_trainable_params'] = param_counts['non_trainable']
        
        logger.info("\n[3/6] Detecting Quantization Level...")
        bits_per_param = get_bits_per_param(self.model)
        results['bits_per_param'] = bits_per_param
        
        logger.info("\n[4/6] Measuring Peak Memory...")
        peak_memory = get_peak_memory(self.use_cuda)
        results['peak_memory_mb'] = peak_memory
        
        logger.info("\n[5/6] Computing Memory Efficiency...")
        memory_efficiency = get_memory_efficiency(size_gb, peak_memory)
        results['memory_efficiency'] = memory_efficiency
        
        logger.info("\n[6/6] Estimating KV Cache Size...")
        try:
            kv_cache_size = estimate_kv_cache_size(
                self.model,
                batch_size=batch_size,
                sequence_length=sequence_length
            )
            results['kv_cache_size_mb'] = kv_cache_size
            results['kv_cache_batch_size'] = batch_size
            results['kv_cache_seq_length'] = sequence_length
        except Exception as e:
            logger.warning(f"KV cache estimation failed: {e}")
            results['kv_cache_size_mb'] = None
        
        logger.info("Space performance summary")
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of results."""
        print(f"\nModel Size:")
        print(f"  Total: {results['model_size_gb']:.3f} GB")
        print(f"  Bytes: {results['model_size_bytes']:,}")
        
        print(f"\nParameters:")
        print(f"  Total: {results['total_params']:,}")
        print(f"  Trainable: {results['trainable_params']:,}")
        print(f"  Non-trainable: {results['non_trainable_params']:,}")
        
        print(f"\nQuantization:")
        print(f"  Bits/param: {results['bits_per_param']:.1f}")
        
        print(f"\nMemory Usage:")
        print(f"  Peak: {results['peak_memory_mb']:.2f} MB")
        if results['memory_efficiency']:
            print(f"  Efficiency: {results['memory_efficiency']:.3f}")
        
        if results['kv_cache_size_mb']:
            print(f"\nKV Cache (batch={results['kv_cache_batch_size']}, seq={results['kv_cache_seq_length']}):")
            print(f"  Size: {results['kv_cache_size_mb']:.2f} MB")
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save results to JSON."""
        output_data = {
            'evaluation_type': 'space_performance',
            'model': str(self.model.model_path),
            'results': results
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=float)
        
        logger.info(f"\nResults saved to: {output_path}")