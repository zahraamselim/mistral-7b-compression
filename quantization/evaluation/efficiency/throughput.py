"""
Throughput measurement utilities.

Measures token generation throughput in tokens per second.
"""

import time
import logging
from typing import List, Dict

import torch
import numpy as np

from models.model_interface import ModelInterface, GenerationConfig
from efficiency.latency import inference_mode

logger = logging.getLogger(__name__)


def measure_throughput(
    model: ModelInterface,
    prompts: List[str],
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """
    Measure generation throughput.
    
    Args:
        model: ModelInterface instance
        prompts: List of prompts for benchmarking
        num_runs: Number of measurement iterations
        max_new_tokens: Maximum tokens to generate per prompt
        
    Returns:
        Dictionary with throughput metrics:
        - throughput: Average tokens per second
        - throughput_std: Standard deviation
        - total_tokens: Total tokens generated
        - total_time: Total time taken
    """
    logger.info(f"Measuring throughput ({num_runs} runs)")
    
    use_cuda = 'cuda' in str(model.device).lower()
    
    # Warmup
    logger.debug("Running throughput warmup")
    config = GenerationConfig(max_new_tokens=10, do_sample=False)
    
    try:
        with inference_mode(use_cuda):
            _ = model.generate(prompts[0], config)
    except Exception as e:
        logger.warning(f"Throughput warmup failed: {e}")
    
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measurement
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    
    total_tokens = 0
    total_time = 0
    per_run_throughputs = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        
        try:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(use_cuda):
                output = model.generate(prompt, config)
            
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            run_time = time.perf_counter() - start_time
            tokens = output.num_generated_tokens
            
            total_tokens += tokens
            total_time += run_time
            
            run_throughput = tokens / run_time if run_time > 0 else 0.0
            per_run_throughputs.append(run_throughput)
            
            logger.debug(
                f"Run {i+1}/{num_runs}: "
                f"{tokens} tokens in {run_time:.3f}s = {run_throughput:.2f} tok/s"
            )
            
        except Exception as e:
            logger.warning(f"Throughput measurement {i} failed: {e}")
            continue
    
    if total_time == 0 or total_tokens == 0:
        logger.error("All throughput measurements failed")
        return {
            'throughput': 0.0,
            'throughput_std': 0.0,
            'total_tokens': 0,
            'total_time': 0.0
        }
    
    avg_throughput = total_tokens / total_time
    throughput_std = np.std(per_run_throughputs) if per_run_throughputs else 0.0
    
    logger.info(f"Throughput: {avg_throughput:.2f} +/- {throughput_std:.2f} tokens/s")
    logger.info(f"Total: {total_tokens} tokens in {total_time:.2f}s")
    
    return {
        'throughput': avg_throughput,
        'throughput_std': throughput_std,
        'total_tokens': total_tokens,
        'total_time': total_time
    }
