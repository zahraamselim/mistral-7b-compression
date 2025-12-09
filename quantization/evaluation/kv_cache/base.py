"""
Abstract base class for KV cache quantizers.

Defines the interface that all quantization methods must implement.
"""

import torch
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass

from .config import KVCacheConfig

logger = logging.getLogger(__name__)


@dataclass
class QuantizedKVCache:
    """Container for quantized KV cache data."""
    
    # Quantized data
    q_keys: torch.Tensor
    q_values: torch.Tensor
    
    # Quantization parameters
    key_scales: Optional[torch.Tensor] = None
    value_scales: Optional[torch.Tensor] = None
    key_zero_points: Optional[torch.Tensor] = None
    value_zero_points: Optional[torch.Tensor] = None
    
    # Outlier data (for dense-sparse methods)
    key_outliers: Optional[torch.Tensor] = None
    value_outliers: Optional[torch.Tensor] = None
    key_outlier_mask: Optional[torch.Tensor] = None
    value_outlier_mask: Optional[torch.Tensor] = None
    
    # Metadata
    is_quantized: bool = True
    layer_idx: int = -1
    position: int = -1
    
    # Codebooks (for non-uniform quantization)
    key_codebook: Optional[torch.Tensor] = None
    value_codebook: Optional[torch.Tensor] = None


class KVCacheQuantizer(ABC):
    """
    Abstract base class for KV cache quantization methods.
    
    All quantization methods must inherit from this class and implement
    the required methods. This ensures consistent interface across different
    quantization approaches.
    """
    
    def __init__(self, config: KVCacheConfig):
        """
        Initialize quantizer.
        
        Args:
            config: Configuration for this quantization method
        """
        self.config = config
        self.cache_store: Dict[int, List[QuantizedKVCache]] = {}
        self.is_calibrated = False
        
        # Statistics for analysis
        self.stats = {
            'total_tokens_processed': 0,
            'total_bytes_original': 0,
            'total_bytes_quantized': 0,
            'num_quantize_calls': 0,
            'num_dequantize_calls': 0,
        }
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def quantize_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> QuantizedKVCache:
        """
        Quantize key-value tensors.
        
        Args:
            keys: Key tensor [batch, num_heads, seq_len, head_dim]
            values: Value tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Index of the transformer layer
            position: Current token position in sequence
            
        Returns:
            QuantizedKVCache containing all quantization data
        """
        pass
    
    @abstractmethod
    def dequantize_kv(
        self,
        quantized_cache: QuantizedKVCache
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantize key-value tensors back to floating point.
        
        Args:
            quantized_cache: QuantizedKVCache instance
            
        Returns:
            Tuple of (keys, values) in original precision
        """
        pass
    
    def calibrate(
        self,
        calibration_data: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Calibrate quantization parameters using calibration data.
        
        Optional method - some quantizers don't need calibration (e.g., KIVI).
        
        Args:
            calibration_data: List of dicts with 'keys' and 'values' tensors
        """
        if not self.config.use_calibration:
            logger.info(f"{self.__class__.__name__} does not use calibration")
            return
        
        logger.info(f"Calibrating {self.__class__.__name__}...")
        self._calibrate_impl(calibration_data)
        self.is_calibrated = True
        logger.info("Calibration complete")
    
    def _calibrate_impl(
        self,
        calibration_data: List[Dict[str, torch.Tensor]]
    ) -> None:
        """
        Implementation of calibration logic.
        
        Override this in subclasses that need calibration.
        """
        pass
    
    def update_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> None:
        """
        Quantize and store KV cache for a layer.
        
        Args:
            keys: Key tensor to quantize and cache
            values: Value tensor to quantize and cache
            layer_idx: Layer index
            position: Current position in sequence
        """
        if layer_idx not in self.cache_store:
            self.cache_store[layer_idx] = []
        
        # Quantize
        quantized = self.quantize_kv(keys, values, layer_idx, position)
        
        # Store
        self.cache_store[layer_idx].append(quantized)
        
        # Update stats
        self.stats['num_quantize_calls'] += 1
        self.stats['total_tokens_processed'] += keys.shape[2]  # seq_len dimension
    
    def get_cache(
        self,
        layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Retrieve and dequantize cached KV for a layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Tuple of (keys, values) or (None, None) if cache is empty
        """
        if layer_idx not in self.cache_store or not self.cache_store[layer_idx]:
            return None, None
        
        all_keys = []
        all_values = []
        
        # Dequantize each cache entry
        for quantized_cache in self.cache_store[layer_idx]:
            keys, values = self.dequantize_kv(quantized_cache)
            all_keys.append(keys)
            all_values.append(values)
        
        # Concatenate along sequence dimension
        keys = torch.cat(all_keys, dim=2)
        values = torch.cat(all_values, dim=2)
        
        # Update stats
        self.stats['num_dequantize_calls'] += 1
        
        return keys, values
    
    def clear_cache(self, layer_idx: Optional[int] = None) -> None:
        """
        Clear cached data.
        
        Args:
            layer_idx: If specified, only clear this layer. Otherwise clear all.
        """
        if layer_idx is not None:
            if layer_idx in self.cache_store:
                self.cache_store[layer_idx].clear()
        else:
            self.cache_store.clear()
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Calculate compression statistics.
        
        Returns:
            Dictionary with compression metrics
        """
        total_quantized_bytes = 0
        total_original_bytes = 0
        num_layers = 0
        total_entries = 0
        
        for layer_cache in self.cache_store.values():
            num_layers += 1
            for cache_entry in layer_cache:
                total_entries += 1
                
                # Estimate quantized size
                q_bytes = self._estimate_quantized_size(cache_entry)
                total_quantized_bytes += q_bytes
                
                # Estimate original size (FP16)
                orig_bytes = self._estimate_original_size(cache_entry)
                total_original_bytes += orig_bytes
        
        compression_ratio = (
            total_original_bytes / total_quantized_bytes 
            if total_quantized_bytes > 0 else 0.0
        )
        
        return {
            'total_original_mb': total_original_bytes / (1024**2),
            'total_quantized_mb': total_quantized_bytes / (1024**2),
            'savings_mb': (total_original_bytes - total_quantized_bytes) / (1024**2),
            'compression_ratio': compression_ratio,
            'num_layers': num_layers,
            'total_cache_entries': total_entries,
            **self.stats
        }
    
    def _estimate_quantized_size(self, cache: QuantizedKVCache) -> int:
        """Estimate size of quantized cache entry in bytes."""
        total_bytes = 0
        
        # Quantized tensors
        if cache.q_keys is not None:
            total_bytes += cache.q_keys.element_size() * cache.q_keys.numel()
        if cache.q_values is not None:
            total_bytes += cache.q_values.element_size() * cache.q_values.numel()
        
        # Scales and zero points (FP16)
        for tensor in [cache.key_scales, cache.value_scales, 
                      cache.key_zero_points, cache.value_zero_points]:
            if tensor is not None:
                total_bytes += tensor.element_size() * tensor.numel()
        
        # Outliers (FP16)
        for tensor in [cache.key_outliers, cache.value_outliers]:
            if tensor is not None:
                total_bytes += tensor.element_size() * tensor.numel()
        
        # Codebooks (FP16)
        for tensor in [cache.key_codebook, cache.value_codebook]:
            if tensor is not None:
                total_bytes += tensor.element_size() * tensor.numel()
        
        return total_bytes
    
    def _estimate_original_size(self, cache: QuantizedKVCache) -> int:
        """Estimate size if cache was stored in FP16."""
        # Assume FP16 (2 bytes per element)
        num_elements = cache.q_keys.numel() + cache.q_values.numel()
        return num_elements * 2  # 2 bytes for FP16
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'total_tokens_processed': 0,
            'total_bytes_original': 0,
            'total_bytes_quantized': 0,
            'num_quantize_calls': 0,
            'num_dequantize_calls': 0,
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"key_bits={self.config.key_bits}, "
            f"value_bits={self.config.value_bits}, "
            f"calibrated={self.is_calibrated})"
        )
