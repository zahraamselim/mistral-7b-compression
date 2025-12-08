"""Throughput measurement utilities."""

import time
import logging
from typing import List, Dict

import torch
import numpy as np

from benchmarks.efficiency.latency import inference_mode

logger = logging.getLogger(__name__)


def measure_throughput(
    model_interface,
    prompts: List[str],
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """
    Measure generation throughput.
    
    Args:
        model_interface: ModelInterface instance
        prompts: Prompts for benchmarking
        num_runs: Number of measurement iterations
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with throughput metrics
    """
    logger.info(f"Measuring throughput ({num_runs} runs)...")
    
    model = model_interface.get_model()
    tokenizer = model_interface.get_tokenizer()
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    # Warmup
    logger.debug("Running throughput warmup...")
    try:
        inputs = tokenizer(prompts[0], return_tensors='pt').to(device)
        with inference_mode(is_cuda):
            _ = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
    except Exception as e:
        logger.warning(f"Throughput warmup failed: {e}")
    
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    total_tokens = 0
    total_time = 0
    per_run_throughputs = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        
        try:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
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
            
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            run_time = end_time - start_time
            
            total_tokens += tokens
            total_time += run_time
            
            run_throughput = tokens / run_time if run_time > 0 else 0.0
            per_run_throughputs.append(run_throughput)
            
            logger.debug(f"Run {i+1}/{num_runs}: {tokens} tokens in {run_time:.3f}s = {run_throughput:.2f} tok/s")
            
        except Exception as e:
            logger.warning(f"Throughput measurement {i+1} failed: {e}")
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


def measure_batch_throughput(
    model_interface,
    prompts: List[str],
    batch_sizes: List[int] = [1, 2, 4, 8],
    max_new_tokens: int = 128
) -> Dict[int, Dict[str, float]]:
    """
    Measure throughput at different batch sizes.
    
    Args:
        model_interface: ModelInterface instance
        prompts: Prompts to use
        batch_sizes: Batch sizes to test
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary mapping batch size to throughput metrics
    """
    logger.info(f"Measuring batch throughput for sizes: {batch_sizes}")
    
    model = model_interface.get_model()
    tokenizer = model_interface.get_tokenizer()
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size {batch_size}...")
        
        try:
            batch_prompts = prompts[:batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Warmup
            with inference_mode(is_cuda):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Measure
            start_time = time.perf_counter()
            
            with inference_mode(is_cuda):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Calculate total tokens
            total_tokens = 0
            for i in range(outputs.shape[0]):
                num_new_tokens = outputs[i].shape[0] - inputs['input_ids'][i].shape[0]
                total_tokens += num_new_tokens
            
            run_time = end_time - start_time
            throughput = total_tokens / run_time if run_time > 0 else 0.0
            
            results[batch_size] = {
                'throughput': throughput,
                'tokens_per_sequence': total_tokens / batch_size,
                'time_seconds': run_time,
                'total_tokens': total_tokens
            }
            
            logger.info(f"Batch size {batch_size}: {throughput:.2f} tok/s ({total_tokens} tokens in {run_time:.2f}s)")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch size {batch_size}, stopping batch tests")
                break
            else:
                logger.error(f"Batch size {batch_size} failed: {e}")
                results[batch_size] = {
                    'throughput': 0.0,
                    'error': str(e)
                }
        except Exception as e:
            logger.error(f"Batch size {batch_size} failed: {e}")
            results[batch_size] = {
                'throughput': 0.0,
                'error': str(e)
            }
    
    return results