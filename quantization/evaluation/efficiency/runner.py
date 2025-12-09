"""
Efficiency benchmark orchestrator.

Coordinates all efficiency measurements and produces comprehensive results.
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from model_interface import ModelInterface
from efficiency.latency import measure_latency, measure_ttft, measure_prefill_decode
from efficiency.throughput import measure_throughput
from efficiency.batch import measure_batch_latency, find_optimal_batch_size
from efficiency.memory import (
    get_model_size, get_parameter_count, get_bits_per_param,
    get_peak_memory, get_memory_efficiency, reset_memory_stats,
    estimate_kv_cache_size
)

from efficiency.energy import estimate_energy
from efficiency.device import get_device_specs, detect_tdp, detect_peak_tflops

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyResults:
    """Results from efficiency benchmarks."""
    
    # Device info
    device: str
    device_name: Optional[str] = None
    
    # Latency metrics
    latency_ms_per_token: float = 0.0
    latency_std: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    ttft_ms: float = 0.0
    ttft_std: float = 0.0
    ttft_min: float = 0.0
    ttft_max: float = 0.0
    prefill_ms: Optional[float] = None
    decode_ms_per_token: Optional[float] = None
    
    # Throughput metrics
    throughput_tokens_per_sec: float = 0.0
    throughput_std: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    model_size_gb: float = 0.0
    total_params: int = 0
    bits_per_param: Optional[float] = None
    memory_efficiency: Optional[float] = None
    kv_cache_size_mb: Optional[float] = None
    

    
    # Energy metrics
    energy_per_token_mj: Optional[float] = None
    tdp_watts: Optional[float] = None
    
    # Comparison metrics (vs baseline)
    compression_ratio: Optional[float] = None
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None
    
    # Batch metrics (for edge deployment)
    batch_results: Optional[Dict[int, Dict[str, float]]] = None
    optimal_batch_size: Optional[int] = None
    optimal_batch_throughput: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return asdict(self)


class EfficiencyBenchmark:
    """
    Comprehensive efficiency benchmark suite.
    
    Measures:
    - Latency (ms/token) with statistics
    - Time to First Token (TTFT)
    - Prefill vs Decode latency
    - Throughput (tokens/second)
    - Memory usage (peak, model size, KV cache)
    - Model parameters and bits per parameter
    - FLOPs and Model FLOPs Utilization (MFU)
    - Energy consumption estimates
    """
    
    def __init__(
        self,
        model: ModelInterface,
        prompts: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """
        Initialize efficiency benchmark.
        
        Args:
            model: ModelInterface instance
            prompts: List of prompts for benchmarking
            verbose: Enable verbose logging
        """
        self.model = model
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Default prompts if none provided
        self.prompts = prompts or [
            "The capital of France is",
            "Artificial intelligence is defined as",
            "In machine learning, the term overfitting refers to",
            "Quantum computing differs from classical computing because",
            "The theory of relativity states that",
            "Natural language processing is",
            "Deep learning models are characterized by",
            "The Transformer architecture introduced",
            "Reinforcement learning agents learn by",
            "Neural networks consist of"
        ]
        
        # Device detection
        self.use_cuda = 'cuda' in str(model.device).lower()
        self.device_specs = get_device_specs(self.use_cuda)
        self.tdp_watts = detect_tdp(self.use_cuda)
        self.peak_tflops = detect_peak_tflops(self.use_cuda)
        
        logger.info("Efficiency Benchmark initialized")
        logger.info(f"  Device: {model.device} ({self.device_specs.get('device_name', 'unknown')})")
        logger.info(f"  TDP: {self.tdp_watts}W")
        logger.info(f"  Peak: {self.peak_tflops} TFLOPs")
    
    def run_all(
        self,
        num_warmup: int = 3,
        num_runs: int = 10,
        max_new_tokens: int = 128,
        measure_prefill_decode: bool = False,
        measure_batch: bool = True,
        batch_sizes: List[int] = [1, 2, 4, 8],
        base_results: Optional[Dict[str, Any]] = None
    ) -> EfficiencyResults:
        """
        Run all efficiency benchmarks.
        
        Args:
            num_warmup: Number of warmup iterations
            num_runs: Number of measurement iterations
            max_new_tokens: Maximum tokens to generate
            measure_prefill_decode: Whether to measure prefill/decode separately
            measure_batch: Whether to measure batch inference (for edge deployment)
            batch_sizes: Batch sizes to test if measure_batch=True
            base_results: Results from baseline model for comparison
            
        Returns:
            EfficiencyResults object with all metrics
        """
        logger.info("Starting efficiency benchmarks")
        logger.info(f"Warmup runs: {num_warmup}")
        logger.info(f"Measurement runs: {num_runs}")
        logger.info(f"Max new tokens: {max_new_tokens}")
        logger.info(f"Prompts: {len(self.prompts)}")
        
        # Reset memory stats
        if self.use_cuda:
            reset_memory_stats(self.use_cuda)
        
        # Static metrics
        logger.info("\n--- Static Metrics ---")
        size_gb, _ = get_model_size(self.model)
        param_counts = get_parameter_count(self.model)
        bits_per_param = get_bits_per_param(self.model)
        
        # Latency metrics
        logger.info("\n--- Latency Metrics ---")
        latency_results = measure_latency(
            self.model,
            self.prompts,
            num_warmup,
            num_runs,
            max_new_tokens
        )
        
        # TTFT metrics
        logger.info("\n--- TTFT Metrics ---")
        ttft_results = measure_ttft(
            self.model,
            self.prompts[0],
            num_runs
        )
        
        # Prefill/Decode metrics (optional)
        prefill_decode_results = None
        if measure_prefill_decode:
            logger.info("\n--- Prefill/Decode Metrics ---")
            try:
                prefill_decode_results = measure_prefill_decode(
                    self.model,
                    self.prompts[0],
                    num_decode_tokens=max_new_tokens // 2,
                    num_runs=min(5, num_runs)
                )
            except Exception as e:
                logger.warning(f"Prefill/decode measurement failed: {e}")
        
        # Throughput metrics
        logger.info("\n--- Throughput Metrics ---")
        throughput_results = measure_throughput(
            self.model,
            self.prompts,
            num_runs,
            max_new_tokens
        )
        
        # Memory metrics
        logger.info("\n--- Memory Metrics ---")
        peak_memory = get_peak_memory(self.use_cuda)
        memory_efficiency = get_memory_efficiency(size_gb, peak_memory)
        
        try:
            kv_cache_size = estimate_kv_cache_size(
                self.model,
                batch_size=1,
                sequence_length=2048
            )
        except Exception as e:
            logger.warning(f"KV cache estimation failed: {e}")
            kv_cache_size = None
        
        # Compute metrics (skip - not directly relevant for quantization/RAG evaluation)
        flops = None
        mfu = None
        
        # Energy metrics
        logger.info("\n--- Energy Metrics ---")
        energy = estimate_energy(latency_results['ms_per_token'], self.tdp_watts)
        
        # Batch metrics (for edge deployment / RAG systems)
        batch_results_data = None
        optimal_batch_size = None
        optimal_batch_throughput = None
        
        if measure_batch and len(self.prompts) >= max(batch_sizes):
            logger.info("\n--- Batch Inference Metrics ---")
            try:
                batch_results_data = measure_batch_latency(
                    self.model,
                    self.prompts,
                    batch_sizes=batch_sizes,
                    max_new_tokens=max_new_tokens // 2,
                    num_runs=min(5, num_runs)
                )
                
                # Find optimal batch size for edge deployment
                optimal_config = find_optimal_batch_size(
                    batch_results_data,
                    latency_constraint_ms=100.0  # 100ms constraint for responsive systems
                )
                optimal_batch_size = optimal_config['optimal_batch_size']
                optimal_batch_throughput = optimal_config['throughput_at_optimal']
                
            except Exception as e:
                logger.warning(f"Batch measurement failed: {e}")
        
        # Comparison metrics
        comparison_metrics = {}
        if base_results:
            logger.info("\n--- Comparison Metrics ---")
            comparison_metrics = self._compute_comparison_metrics(
                base_results,
                size_gb,
                latency_results['ms_per_token'],
                peak_memory
            )
        
        # Create results
        results = EfficiencyResults(
            # Device
            device=str(self.model.device),
            device_name=self.device_specs.get('device_name'),
            
            # Latency
            latency_ms_per_token=latency_results['ms_per_token'],
            latency_std=latency_results.get('latency_std', 0.0),
            latency_min=latency_results.get('latency_min', 0.0),
            latency_max=latency_results.get('latency_max', 0.0),
            ttft_ms=ttft_results['ttft_ms'],
            ttft_std=ttft_results.get('ttft_std', 0.0),
            ttft_min=ttft_results.get('ttft_min', 0.0),
            ttft_max=ttft_results.get('ttft_max', 0.0),
            prefill_ms=prefill_decode_results.get('prefill_ms') if prefill_decode_results else None,
            decode_ms_per_token=prefill_decode_results.get('decode_ms_per_token') if prefill_decode_results else None,
            
            # Throughput
            throughput_tokens_per_sec=throughput_results['throughput'],
            throughput_std=throughput_results.get('throughput_std', 0.0),
            
            # Memory
            peak_memory_mb=peak_memory,
            model_size_gb=size_gb,
            total_params=param_counts['total'],
            bits_per_param=bits_per_param,
            memory_efficiency=memory_efficiency,
            kv_cache_size_mb=kv_cache_size,
            

            
            # Energy
            energy_per_token_mj=energy,
            tdp_watts=self.tdp_watts,
            
            # Comparison
            **comparison_metrics,
            
            # Batch metrics
            batch_results=batch_results_data,
            optimal_batch_size=optimal_batch_size,
            optimal_batch_throughput=optimal_batch_throughput
        )
        
        logger.info("Efficiency benchmarks complete")
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _compute_comparison_metrics(
        self,
        base_results: Dict[str, Any],
        size_gb: float,
        latency: float,
        peak_memory: float
    ) -> Dict[str, Optional[float]]:
        """
        Compute comparison metrics against baseline.
        
        Args:
            base_results: Baseline results dictionary
            size_gb: Current model size in GB
            latency: Current latency in ms/token
            peak_memory: Current peak memory in MB
            
        Returns:
            Dictionary with comparison metrics
        """
        metrics = {
            'compression_ratio': None,
            'speedup': None,
            'memory_reduction': None
        }
        
        # Model size compression
        if 'model_size_gb' in base_results and base_results['model_size_gb'] and size_gb > 0:
            metrics['compression_ratio'] = base_results['model_size_gb'] / size_gb
            logger.info(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        
        # Latency speedup
        if 'latency_ms_per_token' in base_results and base_results['latency_ms_per_token'] and latency > 0:
            metrics['speedup'] = base_results['latency_ms_per_token'] / latency
            logger.info(f"Speedup: {metrics['speedup']:.2f}x")
        
        # Memory reduction
        if 'peak_memory_mb' in base_results and base_results['peak_memory_mb'] and peak_memory > 0:
            metrics['memory_reduction'] = base_results['peak_memory_mb'] / peak_memory
            logger.info(f"Memory reduction: {metrics['memory_reduction']:.2f}x")
        
        return metrics
    
    def _print_summary(self, results: EfficiencyResults) -> None:
        """Print summary of results."""
        print("EFFICIENCY RESULTS SUMMARY")
        
        print("\nLatency:")
        print(f"  {results.latency_ms_per_token:.3f} +/- {results.latency_std:.3f} ms/token")
        print(f"  Range: [{results.latency_min:.3f}, {results.latency_max:.3f}]")
        
        print("\nTTFT:")
        print(f"  {results.ttft_ms:.3f} +/- {results.ttft_std:.3f} ms")
        
        if results.prefill_ms:
            print("\nPrefill/Decode:")
            print(f"  Prefill: {results.prefill_ms:.3f} ms")
            print(f"  Decode: {results.decode_ms_per_token:.3f} ms/token")
        
        print("\nThroughput:")
        print(f"  {results.throughput_tokens_per_sec:.2f} +/- {results.throughput_std:.2f} tokens/s")
        
        print("\nMemory:")
        print(f"  Model size: {results.model_size_gb:.3f} GB")
        print(f"  Peak memory: {results.peak_memory_mb:.2f} MB")
        print(f"  Parameters: {results.total_params:,}")
        if results.bits_per_param:
            print(f"  Bits/param: {results.bits_per_param:.1f}")
        

        
        if results.energy_per_token_mj:
            print("\nEnergy:")
            print(f"  {results.energy_per_token_mj:.3f} mJ/token")
        
        if results.batch_results:
            print("\nBatch Inference (Edge Deployment):")
            for batch_size, metrics in sorted(results.batch_results.items()):
                if 'error' not in metrics:
                    print(f"  Batch {batch_size}: {metrics['latency_per_sample_ms']:.2f}ms/sample, "
                          f"{metrics['throughput_samples_per_sec']:.2f} samples/s")
            if results.optimal_batch_size:
                print(f"  Optimal: batch_size={results.optimal_batch_size} "
                      f"({results.optimal_batch_throughput:.2f} samples/s)")
        
        if results.compression_ratio or results.speedup or results.memory_reduction:
            print("\nComparison vs Baseline:")
            if results.compression_ratio:
                print(f"  Compression: {results.compression_ratio:.2f}x")
            if results.speedup:
                print(f"  Speedup: {results.speedup:.2f}x")
            if results.memory_reduction:
                print(f"  Memory reduction: {results.memory_reduction:.2f}x")
        
    def save_results(self, results: EfficiencyResults, output_path: Path) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: EfficiencyResults object
            output_path: Path to output file
        """
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=float)
        
        logger.info(f"Results saved to {output_path}")