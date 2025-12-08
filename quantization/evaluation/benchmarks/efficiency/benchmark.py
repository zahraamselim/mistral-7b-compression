"""Efficiency benchmark orchestrator."""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from core.benchmark import Benchmark
from core.result import BenchmarkResult
from benchmarks.efficiency.latency import (
    measure_latency,
    measure_ttft,
    measure_prefill_decode_latency,
    measure_loading_time
)
from benchmarks.efficiency.throughput import measure_throughput, measure_batch_throughput
from benchmarks.efficiency.memory import (
    get_peak_memory,
    get_model_size,
    get_bits_per_param,
    get_parameter_count,
    get_memory_efficiency,
    reset_memory_stats,
    estimate_kv_cache_size
)
from benchmarks.efficiency.compute import estimate_flops, calculate_mfu
from benchmarks.efficiency.energy import estimate_energy
from benchmarks.efficiency.device import get_device_specs, get_tdp, get_peak_tflops

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyResult(BenchmarkResult):
    """Results from efficiency benchmarks."""
    
    # Device info
    device: str = None
    device_name: Optional[str] = None
    
    # Loading metrics
    loading_time_seconds: Optional[float] = None
    
    # Latency metrics
    latency_ms_per_token: float = 0.0
    latency_std: float = 0.0
    ttft_ms: float = 0.0
    ttft_std: float = 0.0
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
    
    # Compute metrics
    flops_per_token_gflops: Optional[float] = None
    mfu_percent: Optional[float] = None
    
    # Energy metrics
    energy_per_token_mj: Optional[float] = None
    tdp_watts: Optional[float] = None
    
    # Comparison metrics
    compression_ratio: Optional[float] = None
    speedup: Optional[float] = None
    memory_reduction: Optional[float] = None
    
    # Batch throughput
    batch_throughput: Optional[Dict[int, Dict[str, float]]] = field(default_factory=dict)


class EfficiencyBenchmark(Benchmark[EfficiencyResult]):
    """
    Benchmark suite for measuring model efficiency.
    
    Measures:
        - Model loading time
        - Latency (ms/token) with statistics
        - Time to First Token (TTFT)
        - Prefill vs Decode latency
        - Throughput (tokens/second)
        - Batch throughput at different sizes
        - Memory usage (peak, model size, KV cache)
        - Model parameters and bits per parameter
        - FLOPs and Model FLOPs Utilization (MFU)
        - Energy consumption estimates
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        verbose: bool = False
    ):
        """
        Initialize efficiency benchmark.
        
        Args:
            model_interface: ModelInterface instance
            config: Efficiency config
            verbose: Enable verbose logging
        """
        super().__init__(
            model_interface=model_interface,
            config=config,
            verbose=verbose
        )
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        self.device = model_interface.get_device()
        self.is_cuda = 'cuda' in str(self.device).lower()
        
        self.device_name = self._get_device_name()
        
        specs = get_device_specs(self.is_cuda)
        self.tdp_watts = specs['tdp_watts']
        self.peak_tflops = specs['peak_tflops']
        
        logger.info(f"Efficiency benchmark initialized")
        logger.info(f"  Device: {self.device} ({self.device_name})")
        logger.info(f"  TDP: {self.tdp_watts}W")
        logger.info(f"  Peak: {self.peak_tflops} TFLOPs")
    
    def _get_device_name(self) -> str:
        """Get device name."""
        if self.is_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    return torch.cuda.get_device_name(0)
            except:
                pass
        return str(self.device)
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        if not super().validate_config():
            return False
        
        required = ['num_warmup', 'num_runs', 'max_new_tokens', 'prompts']
        missing = [f for f in required if f not in self.config]
        
        if missing:
            logger.warning(f"Config missing fields: {missing}, using defaults")
        
        prompts = self.config.get('prompts', [])
        if not prompts or not isinstance(prompts, list):
            logger.warning("No prompts in config, using defaults")
            self.config['prompts'] = [
                "The capital of France is",
                "Artificial intelligence is defined as",
                "Machine learning models can"
            ]
        
        return True
    
    def run(
        self,
        prompts: Optional[List[str]] = None,
        num_warmup: Optional[int] = None,
        num_runs: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
        measure_batch: bool = False,
        measure_prefill_decode: bool = True,
        measure_loading: bool = False
    ) -> EfficiencyResult:
        """
        Run all efficiency benchmarks.
        
        Args:
            prompts: Prompts for benchmarking
            num_warmup: Number of warmup iterations
            num_runs: Number of measurement iterations
            max_new_tokens: Maximum tokens to generate
            baseline_results: Baseline results for comparison
            measure_batch: Whether to measure batch throughput
            measure_prefill_decode: Whether to measure prefill/decode
            measure_loading: Whether to measure loading time
            
        Returns:
            EfficiencyResult with all metrics
        """
        self.validate_config()
        
        prompts = prompts or self.config.get('prompts', ["The capital of France is"])
        num_warmup = num_warmup if num_warmup is not None else self.config.get('num_warmup', 3)
        num_runs = num_runs if num_runs is not None else self.config.get('num_runs', 10)
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.get('max_new_tokens', 128)
        
        logger.info("="*60)
        logger.info("Starting efficiency benchmarks")
        logger.info("="*60)
        logger.info(f"Warmup runs: {num_warmup}")
        logger.info(f"Measurement runs: {num_runs}")
        logger.info(f"Max new tokens: {max_new_tokens}")
        logger.info(f"Prompts: {len(prompts)}")
        
        if self.is_cuda:
            reset_memory_stats(self.is_cuda)
        
        # Static metrics
        logger.info("\n--- Static Metrics ---")
        size_gb, _ = get_model_size(self.model)
        param_counts = get_parameter_count(self.model)
        bits_per_param = get_bits_per_param(self.model)
        
        # Loading time
        loading_time = None
        if measure_loading:
            logger.info("\n--- Loading Time ---")
            loading_time = measure_loading_time(
                self.model_interface.__class__,
                self.model_interface.model_path,
                num_runs=3
            )
        
        # Latency
        logger.info("\n--- Latency Metrics ---")
        latency_results = measure_latency(
            self.model_interface,
            prompts,
            num_warmup,
            num_runs,
            max_new_tokens
        )
        
        # TTFT
        logger.info("\n--- TTFT Metrics ---")
        ttft_results = measure_ttft(
            self.model_interface,
            prompts[0],
            num_runs
        )
        
        # Prefill/Decode
        prefill_decode_results = None
        if measure_prefill_decode:
            logger.info("\n--- Prefill/Decode Metrics ---")
            try:
                prefill_decode_results = measure_prefill_decode_latency(
                    self.model_interface,
                    prompts[0],
                    num_decode_tokens=max_new_tokens // 2,
                    num_runs=min(5, num_runs)
                )
            except Exception as e:
                logger.warning(f"Prefill/decode measurement failed: {e}")
        
        # Throughput
        logger.info("\n--- Throughput Metrics ---")
        throughput_results = measure_throughput(
            self.model_interface,
            prompts,
            num_runs,
            max_new_tokens
        )
        
        # Batch throughput
        batch_throughput_results = None
        if measure_batch and len(prompts) >= 4:
            logger.info("\n--- Batch Throughput Metrics ---")
            try:
                batch_throughput_results = measure_batch_throughput(
                    self.model_interface,
                    prompts,
                    batch_sizes=[1, 2, 4, 8],
                    max_new_tokens=max_new_tokens // 2
                )
            except Exception as e:
                logger.warning(f"Batch throughput measurement failed: {e}")
        
        # Memory
        logger.info("\n--- Memory Metrics ---")
        peak_memory = get_peak_memory(self.is_cuda)
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
        
        # Compute
        logger.info("\n--- Compute Metrics ---")
        flops = estimate_flops(self.model)
        throughput = throughput_results['throughput']
        mfu = calculate_mfu(flops, throughput, self.is_cuda, self.peak_tflops) if flops else None
        
        # Energy
        logger.info("\n--- Energy Metrics ---")
        energy = estimate_energy(latency_results['ms_per_token'], self.tdp_watts)
        
        # Comparison
        comparison_metrics = {}
        if baseline_results:
            logger.info("\n--- Comparison Metrics ---")
            comparison_metrics = self._compute_comparison_metrics(
                baseline_results,
                size_gb,
                latency_results['ms_per_token'],
                peak_memory
            )
        
        # Create result
        result = EfficiencyResult(
            device=str(self.device),
            device_name=self.device_name,
            loading_time_seconds=loading_time,
            latency_ms_per_token=latency_results['ms_per_token'],
            latency_std=latency_results.get('latency_std', 0.0),
            ttft_ms=ttft_results['ttft_ms'],
            ttft_std=ttft_results.get('ttft_std', 0.0),
            prefill_ms=prefill_decode_results.get('prefill_ms') if prefill_decode_results else None,
            decode_ms_per_token=prefill_decode_results.get('decode_ms_per_token') if prefill_decode_results else None,
            throughput_tokens_per_sec=throughput,
            throughput_std=throughput_results.get('throughput_std', 0.0),
            peak_memory_mb=peak_memory,
            model_size_gb=size_gb,
            total_params=param_counts['total'],
            bits_per_param=bits_per_param,
            memory_efficiency=memory_efficiency,
            kv_cache_size_mb=kv_cache_size,
            flops_per_token_gflops=flops,
            mfu_percent=mfu,
            energy_per_token_mj=energy,
            tdp_watts=self.tdp_watts,
            **comparison_metrics,
            batch_throughput=batch_throughput_results
        )
        
        logger.info("\n" + "="*60)
        logger.info("Efficiency benchmarks complete")
        logger.info("="*60)
        
        if self.verbose:
            print(result)
        
        return result
    
    def _compute_comparison_metrics(
        self,
        baseline_results: Dict[str, Any],
        size_gb: float,
        latency: float,
        peak_memory: float
    ) -> Dict[str, Optional[float]]:
        """Compute comparison metrics against baseline."""
        metrics = {
            'compression_ratio': None,
            'speedup': None,
            'memory_reduction': None
        }
        
        if 'model_size_gb' in baseline_results and baseline_results['model_size_gb'] and size_gb > 0:
            metrics['compression_ratio'] = baseline_results['model_size_gb'] / size_gb
            logger.info(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
        
        if 'latency_ms_per_token' in baseline_results and baseline_results['latency_ms_per_token'] and latency > 0:
            metrics['speedup'] = baseline_results['latency_ms_per_token'] / latency
            logger.info(f"Speedup: {metrics['speedup']:.2f}x")
        
        if 'peak_memory_mb' in baseline_results and baseline_results['peak_memory_mb'] and peak_memory > 0:
            metrics['memory_reduction'] = baseline_results['peak_memory_mb'] / peak_memory
            logger.info(f"Memory reduction: {metrics['memory_reduction']:.2f}x")
        
        return metrics