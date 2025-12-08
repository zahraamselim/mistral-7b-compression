"""Latency measurement utilities."""

import time
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager

import torch
import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def inference_mode(is_cuda: bool):
    """Context manager for optimized inference."""
    with torch.inference_mode():
        if is_cuda and torch.cuda.is_available():
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    yield
            except Exception as e:
                logger.debug(f"Autocast not available: {e}")
                yield
        else:
            yield


def measure_loading_time(
    model_class,
    model_path: str,
    num_runs: int = 3,
    **load_kwargs
) -> float:
    """
    Measure model loading time.
    
    Critical for edge deployment scenarios.
    
    Args:
        model_class: Model class to instantiate
        model_path: Path to model
        num_runs: Number of runs to average
        **load_kwargs: Arguments for load method
        
    Returns:
        Average loading time in seconds
    """
    logger.info(f"Measuring loading time ({num_runs} runs)...")
    
    loading_times = []
    
    for i in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_time = time.perf_counter()
        
        try:
            model_instance = model_class()
            model_instance.load(model_path, **load_kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            loading_time = end_time - start_time
            loading_times.append(loading_time)
            
            logger.debug(f"Run {i+1}/{num_runs}: {loading_time:.2f}s")
            
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Loading run {i+1} failed: {e}")
            continue
    
    if not loading_times:
        logger.error("All loading measurements failed")
        return float('inf')
    
    avg_loading_time = np.mean(loading_times)
    logger.info(f"Average loading time: {avg_loading_time:.2f}s")
    
    return avg_loading_time


def measure_latency(
    model_interface,
    prompts: List[str],
    num_warmup: int = 3,
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """
    Measure generation latency.
    
    Args:
        model_interface: ModelInterface instance
        prompts: Prompts for benchmarking
        num_warmup: Number of warmup iterations
        num_runs: Number of measurement iterations
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with latency metrics
    """
    logger.info(f"Measuring latency ({num_warmup} warmup, {num_runs} runs)...")
    
    model = model_interface.get_model()
    tokenizer = model_interface.get_tokenizer()
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    # Warmup
    logger.debug("Running warmup...")
    for i in range(num_warmup):
        try:
            inputs = tokenizer(prompts[0], return_tensors='pt', padding=True).to(device)
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            logger.debug(f"Warmup {i+1}/{num_warmup} complete")
        except Exception as e:
            logger.warning(f"Warmup {i+1} failed: {e}")
    
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    latencies = []
    tokens_generated = []
    
    # Measurement
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        
        try:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            
            latencies.append(latency_ms)
            tokens_generated.append(num_tokens)
            
            logger.debug(f"Run {i+1}/{num_runs}: {latency_ms:.2f}ms, {num_tokens} tokens")
            
        except Exception as e:
            logger.warning(f"Run {i+1} failed: {e}")
            continue
    
    if not latencies:
        logger.error("All measurements failed")
        return {
            'ms_per_token': float('inf'),
            'avg_tokens_generated': 0.0,
            'latency_std': 0.0,
            'latency_min': float('inf'),
            'latency_max': float('inf')
        }
    
    avg_tokens = np.mean(tokens_generated)
    ms_per_token = np.mean(latencies) / avg_tokens if avg_tokens > 0 else float('inf')
    
    per_token_latencies = [lat / toks for lat, toks in zip(latencies, tokens_generated) if toks > 0]
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
    model_interface,
    prompt: str,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Measure Time To First Token.
    
    Args:
        model_interface: ModelInterface instance
        prompt: Prompt for measurement
        num_runs: Number of runs
        
    Returns:
        Dictionary with TTFT metrics
    """
    logger.info(f"Measuring TTFT ({num_runs} runs)...")
    
    model = model_interface.get_model()
    tokenizer = model_interface.get_tokenizer()
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    # Warmup
    logger.debug("Running TTFT warmup...")
    inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    for i in range(2):
        try:
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            logger.debug(f"TTFT warmup {i+1}/2 complete")
        except Exception as e:
            logger.warning(f"TTFT warmup {i+1} failed: {e}")
    
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    ttfts = []
    
    for i in range(num_runs):
        try:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            ttft = (end_time - start_time) * 1000
            ttfts.append(ttft)
            
            logger.debug(f"TTFT run {i+1}/{num_runs}: {ttft:.2f}ms")
            
        except Exception as e:
            logger.warning(f"TTFT run {i+1} failed: {e}")
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


def measure_prefill_decode_latency(
    model_interface,
    prompt: str,
    num_decode_tokens: int = 50,
    num_runs: int = 5
) -> Dict[str, float]:
    """
    Measure prefill and decode latency separately.
    
    Args:
        model_interface: ModelInterface instance
        prompt: Prompt for measurement
        num_decode_tokens: Number of tokens to generate
        num_runs: Number of runs
        
    Returns:
        Dictionary with prefill and decode metrics
    """
    logger.info("Measuring prefill vs decode latency...")
    
    model = model_interface.get_model()
    tokenizer = model_interface.get_tokenizer()
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    prefill_times = []
    decode_times = []
    
    for i in range(num_runs):
        try:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            prompt_length = inputs['input_ids'].shape[1]
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure prefill
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                with torch.no_grad():
                    _ = model(**inputs)
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            prefill_time = (time.perf_counter() - start_time) * 1000
            prefill_times.append(prefill_time)
            
            # Measure decode
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=num_decode_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            total_time = (time.perf_counter() - start_time) * 1000
            actual_tokens = outputs.shape[1] - prompt_length
            
            decode_time = total_time - prefill_time
            decode_times.append(decode_time / actual_tokens if actual_tokens > 0 else 0)
            
            logger.debug(f"Run {i+1}/{num_runs}: prefill={prefill_time:.2f}ms, decode={decode_time/actual_tokens:.2f}ms/tok")
            
        except Exception as e:
            logger.warning(f"Run {i+1} failed: {e}")
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