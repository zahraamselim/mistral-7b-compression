"""
Space Performance Evaluation

Measures all memory and storage-related metrics: model size, peak memory,
compression ratio, memory efficiency, KV cache size.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from model_interface import ModelInterface
from efficiency.memory import (
    get_model_size,
    get_parameter_count,
    get_bits_per_param,
    get_peak_memory,
    get_memory_efficiency,
    reset_memory_stats,
    estimate_kv_cache_size
)

logger = logging.getLogger(__name__)


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
    - Compression ratio: Size reduction vs baseline
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
        """
        Run all space performance benchmarks.
        
        Args:
            batch_size: Batch size for KV cache estimation
            sequence_length: Sequence length for KV cache estimation
            reset_stats: Whether to reset memory stats before measurement
            
        Returns:
            Dictionary with all space metrics
        """
        logger.info("=" * 60)
        logger.info("SPACE PERFORMANCE EVALUATION")
        logger.info("=" * 60)
        
        # Reset memory statistics
        if reset_stats:
            reset_memory_stats(self.use_cuda)
        
        results = {}
        
        # 1. Model size
        logger.info("\n[1/6] Measuring Model Size...")
        size_gb, size_bytes = get_model_size(self.model)
        results['model_size_gb'] = size_gb
        results['model_size_bytes'] = size_bytes
        
        # 2. Parameter counts
        logger.info("\n[2/6] Counting Parameters...")
        param_counts = get_parameter_count(self.model)
        results['total_params'] = param_counts['total']
        results['trainable_params'] = param_counts['trainable']
        results['non_trainable_params'] = param_counts['non_trainable']
        
        # 3. Bits per parameter
        logger.info("\n[3/6] Detecting Quantization Level...")
        bits_per_param = get_bits_per_param(self.model)
        results['bits_per_param'] = bits_per_param
        
        # 4. Peak memory
        logger.info("\n[4/6] Measuring Peak Memory...")
        peak_memory = get_peak_memory(self.use_cuda)
        results['peak_memory_mb'] = peak_memory
        
        # 5. Memory efficiency
        logger.info("\n[5/6] Computing Memory Efficiency...")
        memory_efficiency = get_memory_efficiency(size_gb, peak_memory)
        results['memory_efficiency'] = memory_efficiency
        
        # 6. KV cache estimation
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
        
        # Compute compression ratio if baseline provided
        results['compression_ratio'] = None
        results['memory_reduction'] = None
        
        logger.info("\n" + "=" * 60)
        logger.info("SPACE PERFORMANCE SUMMARY")
        logger.info("=" * 60)
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
        
        if results['compression_ratio']:
            print(f"\nCompression:")
            print(f"  Ratio: {results['compression_ratio']:.2f}x")
        
        if results['memory_reduction']:
            print(f"  Memory reduction: {results['memory_reduction']:.2f}x")
    
    def compare_with_baseline(
        self,
        results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare results with baseline.
        
        Args:
            results: Current results
            baseline_results: Baseline results
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        # Compression ratio
        if baseline_results['model_size_gb'] > 0 and results['model_size_gb'] > 0:
            comparison['compression_ratio'] = (
                baseline_results['model_size_gb'] / 
                results['model_size_gb']
            )
        
        # Memory reduction
        if baseline_results['peak_memory_mb'] > 0 and results['peak_memory_mb'] > 0:
            comparison['memory_reduction'] = (
                baseline_results['peak_memory_mb'] / 
                results['peak_memory_mb']
            )
        
        # Parameter count change (should be same for quantization)
        comparison['param_count_match'] = (
            results['total_params'] == baseline_results['total_params']
        )
        
        # Bits per param reduction
        if baseline_results['bits_per_param'] > 0:
            comparison['bits_reduction'] = (
                baseline_results['bits_per_param'] / 
                results['bits_per_param']
            )
        
        logger.info(f"\nComparison vs Baseline:")
        logger.info(f"  Compression ratio: {comparison.get('compression_ratio', 0):.2f}x")
        logger.info(f"  Memory reduction: {comparison.get('memory_reduction', 0):.2f}x")
        logger.info(f"  Bits reduction: {comparison.get('bits_reduction', 0):.2f}x")
        logger.info(f"  Parameter count match: {comparison.get('param_count_match', False)}")
        
        return comparison
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path,
        comparison: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save results to JSON.
        
        Args:
            results: Results dictionary
            output_path: Output file path
            comparison: Optional comparison metrics
        """
        output_data = {
            'evaluation_type': 'space_performance',
            'model': str(self.model.model_path),
            'results': results
        }
        
        if comparison:
            output_data['comparison'] = comparison
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=float)
        
        logger.info(f"\nResults saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    """
    Example: Evaluate space performance of a model
    
    Usage:
        from model_interface import HuggingFaceModel
        
        model = HuggingFaceModel("mistralai/Mistral-7B-v0.1")
        model.load()
        
        evaluator = SpacePerformanceEvaluator(model, use_cuda=True, verbose=True)
        results = evaluator.run(
            batch_size=1,
            sequence_length=2048,
            reset_stats=True
        )
        
        evaluator.save_results(
            results,
            Path("results/space_performance.json")
        )
    """
    pass