"""
Time Performance Evaluation

Measures all time-related metrics: latency, throughput, TTFT, prefill/decode timing.
All measurement logic is contained within this file.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import json

import torch
import numpy as np

from model_interface import ModelInterface, GenerationConfig

logger = logging.getLogger(__name__)


@contextmanager
def inference_mode(use_cuda: bool):
    """Context manager for optimized inference."""
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
    """Measure generation latency."""
    logger.info(f"Measuring latency ({num_warmup} warmup, {num_runs} runs)")
    
    use_cuda = 'cuda' in str(model.device).lower()
    
    config = GenerationConfig(max_new_tokens=10, do_sample=False)
    
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
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    
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
    """Measure Time To First Token."""
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


def measure_throughput(
    model: ModelInterface,
    prompts: List[str],
    num_runs: int = 10,
    max_new_tokens: int = 128
) -> Dict[str, float]:
    """Measure generation throughput."""
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


class TimePerformanceEvaluator:
    """
    Evaluates all time-related performance metrics.
    
    Metrics measured:
    - Latency (ms/token): End-to-end generation speed
    - TTFT (ms): Time to first token (responsiveness)
    - Throughput (tokens/s): Overall generation rate
    """
    
    def __init__(
        self,
        model: ModelInterface,
        prompts: Optional[list] = None,
        verbose: bool = False
    ):
        self.model = model
        self.verbose = verbose
        
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
    
    def run(
        self,
        num_warmup: int = 3,
        num_runs: int = 10,
        max_new_tokens: int = 128
    ) -> Dict[str, Any]:
        """Run all time performance benchmarks."""
        logger.info("=" * 60)
        logger.info("TIME PERFORMANCE EVALUATION")
        logger.info("=" * 60)
        
        results = {}
        
        logger.info("\n[1/3] Measuring Latency...")
        latency_results = measure_latency(
            self.model,
            self.prompts,
            num_warmup=num_warmup,
            num_runs=num_runs,
            max_new_tokens=max_new_tokens
        )
        results['latency'] = latency_results
        
        logger.info("\n[2/3] Measuring Time To First Token...")
        ttft_results = measure_ttft(
            self.model,
            self.prompts[0],
            num_runs=num_runs
        )
        results['ttft'] = ttft_results
        
        logger.info("\n[3/3] Measuring Throughput...")
        throughput_results = measure_throughput(
            self.model,
            self.prompts,
            num_runs=num_runs,
            max_new_tokens=max_new_tokens
        )
        results['throughput'] = throughput_results
        
        logger.info("\n" + "=" * 60)
        logger.info("TIME PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of results."""
        lat = results['latency']
        print(f"\nLatency:")
        print(f"  Average: {lat['ms_per_token']:.3f} +/- {lat.get('latency_std', 0):.3f} ms/token")
        print(f"  Range: [{lat.get('latency_min', 0):.3f}, {lat.get('latency_max', 0):.3f}]")
        
        ttft = results['ttft']
        print(f"\nTime To First Token:")
        print(f"  Average: {ttft['ttft_ms']:.3f} +/- {ttft.get('ttft_std', 0):.3f} ms")
        print(f"  Range: [{ttft.get('ttft_min', 0):.3f}, {ttft.get('ttft_max', 0):.3f}]")
        
        tp = results['throughput']
        print(f"\nThroughput:")
        print(f"  Average: {tp['throughput']:.2f} +/- {tp.get('throughput_std', 0):.2f} tokens/s")
        print(f"  Total: {tp['total_tokens']} tokens in {tp['total_time']:.2f}s")
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save results to JSON."""
        output_data = {
            'evaluation_type': 'time_performance',
            'model': str(self.model.model_path),
            'results': results
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=float)
        
        logger.info(f"\nResults saved to: {output_path}")