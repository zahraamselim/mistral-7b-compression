"""
Batch inference measurement utilities.

Critical for edge deployment and RAG systems that process multiple queries.
Measures how efficiently models handle concurrent requests.
"""

import time
import logging
from typing import List, Dict

import torch
import numpy as np

from models.model_interface import ModelInterface, GenerationConfig
from efficiency.latency import inference_mode

logger = logging.getLogger(__name__)


def measure_batch_latency(
    model: ModelInterface,
    prompts: List[str],
    batch_sizes: List[int] = [1, 2, 4, 8],
    max_new_tokens: int = 64,
    num_runs: int = 5
) -> Dict[int, Dict[str, float]]:
    """
    Measure latency at different batch sizes.
    
    Critical for edge deployment: shows how model scales with concurrent requests.
    Helps determine optimal batch size for throughput vs latency tradeoff.
    
    Args:
        model: ModelInterface instance
        prompts: List of prompts for benchmarking
        batch_sizes: List of batch sizes to test
        max_new_tokens: Maximum tokens to generate
        num_runs: Number of measurement iterations per batch size
        
    Returns:
        Dictionary mapping batch size to metrics:
        - latency_ms: Total time for batch
        - latency_per_sample_ms: Average time per sample
        - throughput_samples_per_sec: Samples processed per second
        - total_tokens: Total tokens generated
        - memory_mb: Peak memory for this batch size
    """
    logger.info(f"Measuring batch latency for sizes: {batch_sizes}")
    
    use_cuda = 'cuda' in str(model.device).lower()
    results = {}
    
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    
    for batch_size in batch_sizes:
        if len(prompts) < batch_size:
            logger.warning(f"Not enough prompts for batch size {batch_size}, skipping")
            continue
        
        logger.info(f"Testing batch size {batch_size}")
        
        batch_latencies = []
        batch_tokens = []
        
        try:
            # Reset memory stats for this batch size
            if use_cuda and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            for run in range(num_runs):
                batch_prompts = prompts[:batch_size]
                
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                # Generate for each prompt in batch
                # Note: True batch processing depends on model implementation
                total_tokens = 0
                with inference_mode(use_cuda):
                    for prompt in batch_prompts:
                        output = model.generate(prompt, config)
                        total_tokens += output.num_generated_tokens
                
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                batch_time = (time.perf_counter() - start_time) * 1000
                
                batch_latencies.append(batch_time)
                batch_tokens.append(total_tokens)
                
                logger.debug(
                    f"  Run {run+1}/{num_runs}: {batch_time:.2f}ms, "
                    f"{total_tokens} tokens"
                )
            
            # Calculate statistics
            avg_latency = np.mean(batch_latencies)
            latency_per_sample = avg_latency / batch_size
            avg_tokens = np.mean(batch_tokens)
            throughput = batch_size / (avg_latency / 1000.0)
            
            # Get peak memory for this batch size
            peak_memory_mb = 0.0
            if use_cuda and torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            
            results[batch_size] = {
                'latency_ms': avg_latency,
                'latency_per_sample_ms': latency_per_sample,
                'throughput_samples_per_sec': throughput,
                'total_tokens': int(avg_tokens),
                'memory_mb': peak_memory_mb
            }
            
            logger.info(
                f"Batch size {batch_size}: "
                f"{latency_per_sample:.2f}ms/sample, "
                f"{throughput:.2f} samples/s, "
                f"{peak_memory_mb:.0f}MB"
            )
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch size {batch_size}, stopping")
                results[batch_size] = {'error': 'OOM'}
                break
            else:
                logger.error(f"Batch size {batch_size} failed: {e}")
                results[batch_size] = {'error': str(e)}
        except Exception as e:
            logger.error(f"Batch size {batch_size} failed: {e}")
            results[batch_size] = {'error': str(e)}
    
    return results


def find_optimal_batch_size(
    batch_results: Dict[int, Dict[str, float]],
    latency_constraint_ms: float = 1000.0
) -> Dict[str, any]:
    """
    Find optimal batch size given latency constraint.
    
    Critical for edge deployment: determines best batch size for
    maximizing throughput while meeting latency SLAs.
    
    Args:
        batch_results: Results from measure_batch_latency
        latency_constraint_ms: Maximum acceptable latency per sample
        
    Returns:
        Dictionary with:
        - optimal_batch_size: Best batch size
        - throughput_at_optimal: Throughput at optimal batch size
        - latency_at_optimal: Latency at optimal batch size
        - all_valid_sizes: All batch sizes meeting constraint
    """
    valid_sizes = []
    
    for batch_size, metrics in batch_results.items():
        if 'error' in metrics:
            continue
        
        if metrics['latency_per_sample_ms'] <= latency_constraint_ms:
            valid_sizes.append({
                'batch_size': batch_size,
                'throughput': metrics['throughput_samples_per_sec'],
                'latency': metrics['latency_per_sample_ms']
            })
    
    if not valid_sizes:
        logger.warning("No batch sizes meet latency constraint")
        return {
            'optimal_batch_size': 1,
            'throughput_at_optimal': 0.0,
            'latency_at_optimal': float('inf'),
            'all_valid_sizes': []
        }
    
    # Find batch size with highest throughput
    optimal = max(valid_sizes, key=lambda x: x['throughput'])
    
    logger.info(
        f"Optimal batch size: {optimal['batch_size']} "
        f"(throughput: {optimal['throughput']:.2f} samples/s, "
        f"latency: {optimal['latency']:.2f}ms)"
    )
    
    return {
        'optimal_batch_size': optimal['batch_size'],
        'throughput_at_optimal': optimal['throughput'],
        'latency_at_optimal': optimal['latency'],
        'all_valid_sizes': valid_sizes
    }


def measure_scaling_efficiency(
    batch_results: Dict[int, Dict[str, float]]
) -> Dict[str, float]:
    """
    Measure how efficiently the model scales with batch size.
    
    Useful for understanding parallelization efficiency and memory overhead.
    
    Args:
        batch_results: Results from measure_batch_latency
        
    Returns:
        Dictionary with scaling metrics:
        - linear_scaling_score: How close to linear scaling (1.0 = perfect)
        - memory_scaling_factor: Memory increase per batch size increase
        - throughput_scaling_factor: Throughput increase per batch size increase
    """
    valid_results = {
        k: v for k, v in batch_results.items() 
        if 'error' not in v
    }
    
    if len(valid_results) < 2:
        return {
            'linear_scaling_score': 0.0,
            'memory_scaling_factor': 0.0,
            'throughput_scaling_factor': 0.0
        }
    
    # Sort by batch size
    sorted_results = sorted(valid_results.items(), key=lambda x: x[0])
    
    # Calculate scaling metrics
    batch_sizes = [x[0] for x in sorted_results]
    throughputs = [x[1]['throughput_samples_per_sec'] for x in sorted_results]
    memories = [x[1].get('memory_mb', 0) for x in sorted_results]
    
    # Ideal linear scaling: throughput should scale linearly with batch size
    ideal_throughputs = [throughputs[0] * (bs / batch_sizes[0]) for bs in batch_sizes]
    actual_vs_ideal = [actual / ideal for actual, ideal in zip(throughputs, ideal_throughputs)]
    linear_scaling_score = np.mean(actual_vs_ideal)
    
    # Memory scaling: how much memory increases per batch size unit
    if len(memories) > 1 and memories[-1] > 0:
        memory_scaling = (memories[-1] - memories[0]) / (batch_sizes[-1] - batch_sizes[0])
    else:
        memory_scaling = 0.0
    
    # Throughput scaling: average improvement factor
    if len(throughputs) > 1:
        throughput_ratios = [
            throughputs[i] / throughputs[i-1] 
            for i in range(1, len(throughputs))
        ]
        throughput_scaling = np.mean(throughput_ratios)
    else:
        throughput_scaling = 1.0
    
    logger.info(f"Scaling efficiency: {linear_scaling_score:.2f} (1.0 = perfect linear)")
    logger.info(f"Memory overhead: {memory_scaling:.2f} MB per batch size unit")
    logger.info(f"Throughput scaling: {throughput_scaling:.2f}x per doubling")
    
    return {
        'linear_scaling_score': linear_scaling_score,
        'memory_scaling_factor': memory_scaling,
        'throughput_scaling_factor': throughput_scaling
    }
