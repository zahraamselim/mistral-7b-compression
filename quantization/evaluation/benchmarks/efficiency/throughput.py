"""Throughput measurement utilities."""

import time
import logging
from typing import List, Dict

import torch
import numpy as np

logger = logging.getLogger(__name__)


def _tokenize_safe(model_interface, text, padding: bool = True, max_length: int = None):
    """
    Safely tokenize text handling different tokenizer interfaces.
    
    Args:
        model_interface: ModelInterface instance
        text: Input text (string or list of strings)
        padding: Whether to pad
        max_length: Maximum length
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    tokenizer = model_interface.get_tokenizer()
    device = model_interface.get_device()
    
    if hasattr(model_interface, 'tokenize'):
        if isinstance(text, list):
            all_input_ids = []
            for t in text:
                result = model_interface.tokenize(t, add_special_tokens=True, return_tensors='pt', padding=padding)
                all_input_ids.append(result['input_ids'])
            
            max_len = max(ids.shape[1] for ids in all_input_ids)
            padded_ids = []
            for ids in all_input_ids:
                if ids.shape[1] < max_len:
                    pad_size = max_len - ids.shape[1]
                    padding_tensor = torch.zeros(ids.shape[0], pad_size, dtype=ids.dtype)
                    ids = torch.cat([ids, padding_tensor], dim=1)
                padded_ids.append(ids)
            
            input_ids = torch.cat(padded_ids, dim=0)
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            return model_interface.tokenize(text, add_special_tokens=True, return_tensors='pt', padding=padding)
    
    kwargs = {'return_tensors': 'pt', 'padding': padding}
    if max_length:
        kwargs['truncation'] = True
        kwargs['max_length'] = max_length
    
    return tokenizer(text, **kwargs).to(device)


def inference_mode(is_cuda: bool):
    """Context manager for optimized inference."""
    import contextlib
    
    @contextlib.contextmanager
    def _inner():
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
    
    return _inner()


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
    
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    logger.debug("Running throughput warmup...")
    try:
        _ = model_interface.generate(
            prompts[0],
            max_new_tokens=10,
            do_sample=False
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
            tokenizer = model_interface.get_tokenizer()
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            output = model_interface.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            if isinstance(output, str):
                output_tokens = tokenizer.encode(output, add_bos=False, add_eos=False) if hasattr(tokenizer, 'encode') else tokenizer(output, add_special_tokens=False)['input_ids']
                if hasattr(output_tokens, '__len__'):
                    tokens = len(output_tokens)
                else:
                    tokens = output_tokens.shape[0] if hasattr(output_tokens, 'shape') else max_new_tokens
            else:
                tokens = max_new_tokens
            
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
    
    device = model_interface.get_device()
    is_cuda = 'cuda' in str(device).lower()
    
    results = {}
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size {batch_size}...")
        
        if batch_size > 1:
            logger.warning(f"Batch size {batch_size} > 1 not supported for ExLlamaV2, skipping")
            continue
        
        try:
            prompt = prompts[0]
            
            _ = model_interface.generate(
                prompt,
                max_new_tokens=10,
                do_sample=False
            )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            output = model_interface.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
            if is_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            tokenizer = model_interface.get_tokenizer()
            if isinstance(output, str):
                output_tokens = tokenizer.encode(output, add_bos=False, add_eos=False) if hasattr(tokenizer, 'encode') else tokenizer(output, add_special_tokens=False)['input_ids']
                if hasattr(output_tokens, '__len__'):
                    total_tokens = len(output_tokens)
                else:
                    total_tokens = output_tokens.shape[0] if hasattr(output_tokens, 'shape') else max_new_tokens
            else:
                total_tokens = max_new_tokens
            
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