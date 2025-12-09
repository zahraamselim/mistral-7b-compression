"""
Efficiency evaluation module.

Provides comprehensive efficiency benchmarking for language models including
latency, throughput, memory, energy, and batch inference measurements.
Optimized for quantization evaluation, RAG systems, and edge deployment.
"""

from efficiency.runner import EfficiencyBenchmark, EfficiencyResults
from efficiency.latency import measure_latency, measure_ttft, measure_prefill_decode
from efficiency.throughput import measure_throughput
from efficiency.batch import (
    measure_batch_latency, 
    find_optimal_batch_size,
    measure_scaling_efficiency
)
from efficiency.memory import (
    get_model_size,
    get_parameter_count,
    get_bits_per_param,
    get_peak_memory,
    get_memory_efficiency,
    reset_memory_stats,
    estimate_kv_cache_size
)
from efficiency.energy import (
    estimate_energy,
    estimate_total_energy,
    estimate_energy_cost,
    estimate_carbon_footprint
)
from efficiency.device_specs import get_device_specs, detect_tdp, detect_peak_tflops

__all__ = [
    'EfficiencyBenchmark',
    'EfficiencyResults',
    'measure_latency',
    'measure_ttft',
    'measure_prefill_decode',
    'measure_throughput',
    'measure_batch_latency',
    'find_optimal_batch_size',
    'measure_scaling_efficiency',
    'get_model_size',
    'get_parameter_count',
    'get_bits_per_param',
    'get_peak_memory',
    'get_memory_efficiency',
    'reset_memory_stats',
    'estimate_kv_cache_size',
    'estimate_energy',
    'estimate_total_energy',
    'estimate_energy_cost',
    'estimate_carbon_footprint',
    'get_device_specs',
    'detect_tdp',
    'detect_peak_tflops',
]
