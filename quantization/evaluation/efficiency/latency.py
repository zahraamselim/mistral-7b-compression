"""
Latency measurement utilities.

Measures end-to-end generation latency, time to first token (TTFT),
and prefill vs decode latency for language models.
"""

import time
import logging
from typing import List, Dict
from contextlib import contextmanager

import torch
import numpy as np

from models.model_interface import ModelInterface, GenerationConfig

logger = logging.getLogger(__name__)


@contextmanager
def inference_mode(use_cuda: bool):
    """
    Context manager for optimized inference.
    
    Args:
        use_cuda: Whether CUDA is available
    """
    with torch.inference_mode():
        if use_cuda and torch.cuda.is_available():
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    yield
            except Exception:
                yield
        else:
            yield


def measure_latency(
    model: ModelInterface,
    prompts: List[str],
    num_warmup: int = 3,
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """
    Measure generation latency.
    
    Args:
        model: ModelInterface instance
        prompts: List of prompts for benchmarking
        num_warmup: Number of warmup iterations
        num_runs: Number of measurement iterations
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with latency metrics:
        - ms_per_token: Average milliseconds per token
        - avg_tokens_generated: Average tokens generated
        - latency_std: Standard deviation
        - latency_min: Minimum latency
        - latency_max: Maximum latency
    """
    logger.info(f"Measuring latency ({num_warmup} warmup, {num_runs} runs)")
    
    use_cuda = 'cuda' in str(model.device).lower()
    
    config = GenerationConfig(
        max_new_tokens=10,
        do_sample=False
    )
    
    # Warmup
    logger.debug("Running warmup iterations")
    for i in range(num_warmup):
        try:
            with inference_mode(use_cuda):
                _ = model.generate(prompts[0], config)
        except Exception as e:
            logger.warning(f"Warmup run {i} failed: {e}")
    
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measurement
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    
    latencies = []
    tokens_generated = []
    
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
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            latencies.append(latency_ms)
            tokens_generated.append(output.num_generated_tokens)
            
            logger.debug(f"Run {i+1}/{num_runs}: {latency_ms:.2f}ms, {output.num_generated_tokens} tokens")
            
        except Exception as e:
            logger.warning(f"Measurement run {i} failed: {e}")
            continue
    
    if not latencies:
        logger.error("All measurement runs failed")
        return {
            'ms_per_token': float('inf'),
            'avg_tokens_generated': 0.0,
            'latency_std': 0.0,
            'latency_min': float('inf'),
            'latency_max': float('inf')
        }
    
    # Calculate statistics
    avg_tokens = np.mean(tokens_generated)
    ms_per_token = np.mean(latencies) / avg_tokens if avg_tokens > 0 else float('inf')
    
    per_token_latencies = [
        lat / toks for lat, toks in zip(latencies, tokens_generated) if toks > 0
    ]
    
    latency_std = np.std(per_token_latencies) if per_token_latencies else 0.0
    latency_min = np.min(per_token_latencies) if per_token_latencies else float('inf')
    latency_max = np.max(per_token_latencies) if per_token_latencies else float('inf')
    
    logger.info(f"Latency: {ms_per_token:.3f} +/- {latency_std:.3f} ms/token")
    logger.info(f"Range: [{latency_min:.3f}, {latency_max:.3f}] ms/token")
    
    return {
        'ms_per_token': ms_per_token,
        'avg_tokens_generated': avg_tokens,
        'latency_std': latency_std,
        'latency_min': latency_min,
        'latency_max': latency_max
    }


def measure_ttft(
    model: ModelInterface,
    prompt: str,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Measure Time To First Token.
    
    Args:
        model: ModelInterface instance
        prompt: Prompt for measurement
        num_runs: Number of measurement iterations
        
    Returns:
        Dictionary with TTFT metrics:
        - ttft_ms: Average time to first token in milliseconds
        - ttft_std: Standard deviation
        - ttft_min: Minimum TTFT
        - ttft_max: Maximum TTFT
    """
    logger.info(f"Measuring TTFT ({num_runs} runs)")
    
    use_cuda = 'cuda' in str(model.device).lower()
    
    # Warmup
    logger.debug("Running TTFT warmup")
    config = GenerationConfig(max_new_tokens=1, do_sample=False)
    
    for i in range(2):
        try:
            with inference_mode(use_cuda):
                _ = model.generate(prompt, config)
        except Exception as e:
            logger.warning(f"TTFT warmup {i} failed: {e}")
    
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measurement
    ttfts = []
    
    for i in range(num_runs):
        try:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(use_cuda):
                _ = model.generate(prompt, config)
            
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            ttft = (time.perf_counter() - start_time) * 1000
            ttfts.append(ttft)
            
            logger.debug(f"TTFT run {i+1}/{num_runs}: {ttft:.2f}ms")
            
        except Exception as e:
            logger.warning(f"TTFT measurement {i} failed: {e}")
            continue
    
    if not ttfts:
        logger.error("All TTFT measurements failed")
        return {
            'ttft_ms': float('inf'),
            'ttft_std': 0.0,
            'ttft_min': float('inf'),
            'ttft_max': float('inf')
        }
    
    ttft_ms = np.mean(ttfts)
    ttft_std = np.std(ttfts)
    ttft_min = np.min(ttfts)
    ttft_max = np.max(ttfts)
    
    logger.info(f"TTFT: {ttft_ms:.3f} +/- {ttft_std:.3f} ms")
    logger.info(f"Range: [{ttft_min:.3f}, {ttft_max:.3f}] ms")
    
    return {
        'ttft_ms': ttft_ms,
        'ttft_std': ttft_std,
        'ttft_min': ttft_min,
        'ttft_max': ttft_max
    }


def measure_prefill_decode(
    model: ModelInterface,
    prompt: str,
    num_decode_tokens: int = 50,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Separately measure prefill (prompt processing) and decode (generation) latency.
    
    Note: This requires model.model to support forward() directly.
    May not work with all model implementations.
    
    Args:
        model: ModelInterface instance
        prompt: Prompt for measurement
        num_decode_tokens: Number of tokens to generate
        num_runs: Number of measurement iterations
        
    Returns:
        Dictionary with:
        - prefill_ms: Time to process prompt
        - decode_ms_per_token: Time per generated token
        - prefill_decode_ratio: Ratio of prefill to decode time
    """
    logger.info("Measuring prefill vs decode latency")
    
    use_cuda = 'cuda' in str(model.device).lower()
    
    prefill_times = []
    decode_times = []
    
    for i in range(num_runs):
        try:
            inputs = model.encode(prompt)
            prompt_length = inputs['input_ids'].shape[1]
            
            # Measure prefill
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(use_cuda):
                with torch.no_grad():
                    _ = model.model(**inputs)
            
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            prefill_time = (time.perf_counter() - start_time) * 1000
            prefill_times.append(prefill_time)
            
            # Measure total generation time
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            config = GenerationConfig(
                max_new_tokens=num_decode_tokens,
                do_sample=False
            )
            
            with inference_mode(use_cuda):
                output = model.generate(prompt, config)
            
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_time = (time.perf_counter() - start_time) * 1000
            actual_tokens = output.num_generated_tokens
            
            # Decode time = total - prefill
            decode_time = total_time - prefill_time
            decode_times.append(decode_time / actual_tokens if actual_tokens > 0 else 0)
            
            logger.debug(
                f"Run {i+1}/{num_runs}: "
                f"prefill={prefill_time:.2f}ms, "
                f"decode={decode_time/actual_tokens:.2f}ms/tok"
            )
            
        except Exception as e:
            logger.warning(f"Prefill/decode measurement {i} failed: {e}")
            continue
    
    if not prefill_times or not decode_times:
        logger.error("Prefill/decode measurements failed")
        return {
            'prefill_ms': float('inf'),
            'decode_ms_per_token': float('inf'),
            'prefill_decode_ratio': 0.0
        }
    
    avg_prefill = np.mean(prefill_times)
    avg_decode = np.mean(decode_times)
    ratio = avg_prefill / (avg_decode * num_decode_tokens) if avg_decode > 0 else 0.0
    
    logger.info(f"Prefill: {avg_prefill:.3f}ms, Decode: {avg_decode:.3f}ms/token")
    logger.info(f"Prefill/Decode ratio: {ratio:.2f}")
    
    return {
        'prefill_ms': avg_prefill,
        'decode_ms_per_token': avg_decode,
        'prefill_decode_ratio': ratio
    }
