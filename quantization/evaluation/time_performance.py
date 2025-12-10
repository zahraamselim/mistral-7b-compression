"""
Time Performance Evaluation

Measures all time-related metrics: latency, throughput, TTFT, prefill/decode timing.
Combines latency and throughput as they're complementary speed metrics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from model_interface import ModelInterface
from efficiency.latency import measure_latency, measure_ttft, measure_prefill_decode
from efficiency.throughput import measure_throughput

logger = logging.getLogger(__name__)


class TimePerformanceEvaluator:
    """
    Evaluates all time-related performance metrics.
    
    Metrics measured:
    - Latency (ms/token): End-to-end generation speed
    - TTFT (ms): Time to first token (responsiveness)
    - Prefill (ms): Prompt processing time
    - Decode (ms/token): Per-token generation time
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
        
        # Default prompts covering various lengths
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
        max_new_tokens: int = 128,
        measure_prefill_decode_split: bool = False
    ) -> Dict[str, Any]:
        """
        Run all time performance benchmarks.
        
        Args:
            num_warmup: Warmup iterations
            num_runs: Measurement iterations
            max_new_tokens: Tokens to generate
            measure_prefill_decode_split: Whether to separate prefill/decode
            
        Returns:
            Dictionary with all time metrics
        """
        logger.info("=" * 60)
        logger.info("TIME PERFORMANCE EVALUATION")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. Latency metrics
        logger.info("\n[1/4] Measuring Latency...")
        latency_results = measure_latency(
            self.model,
            self.prompts,
            num_warmup=num_warmup,
            num_runs=num_runs,
            max_new_tokens=max_new_tokens
        )
        results['latency'] = latency_results
        
        # 2. TTFT metrics
        logger.info("\n[2/4] Measuring Time To First Token...")
        ttft_results = measure_ttft(
            self.model,
            self.prompts[0],
            num_runs=num_runs
        )
        results['ttft'] = ttft_results
        
        # 3. Prefill/Decode split (optional)
        if measure_prefill_decode_split:
            logger.info("\n[3/4] Measuring Prefill/Decode Split...")
            try:
                prefill_decode_results = measure_prefill_decode(
                    self.model,
                    self.prompts[0],
                    num_decode_tokens=max_new_tokens // 2,
                    num_runs=min(5, num_runs)
                )
                results['prefill_decode'] = prefill_decode_results
            except Exception as e:
                logger.warning(f"Prefill/decode measurement failed: {e}")
                results['prefill_decode'] = None
        else:
            logger.info("\n[3/4] Skipping Prefill/Decode Split")
            results['prefill_decode'] = None
        
        # 4. Throughput metrics
        logger.info("\n[4/4] Measuring Throughput...")
        throughput_results = measure_throughput(
            self.model,
            self.prompts,
            num_runs=num_runs,
            max_new_tokens=max_new_tokens
        )
        results['throughput'] = throughput_results
        
        # Compute speedup if baseline provided
        results['speedup_vs_baseline'] = None
        
        logger.info("\n" + "=" * 60)
        logger.info("TIME PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of results."""
        
        # Latency
        lat = results['latency']
        print(f"\nLatency:")
        print(f"  Average: {lat['ms_per_token']:.3f} ± {lat.get('latency_std', 0):.3f} ms/token")
        print(f"  Range: [{lat.get('latency_min', 0):.3f}, {lat.get('latency_max', 0):.3f}]")
        
        # TTFT
        ttft = results['ttft']
        print(f"\nTime To First Token:")
        print(f"  Average: {ttft['ttft_ms']:.3f} ± {ttft.get('ttft_std', 0):.3f} ms")
        print(f"  Range: [{ttft.get('ttft_min', 0):.3f}, {ttft.get('ttft_max', 0):.3f}]")
        
        # Prefill/Decode
        if results['prefill_decode']:
            pd = results['prefill_decode']
            print(f"\nPrefill/Decode Split:")
            print(f"  Prefill: {pd['prefill_ms']:.3f} ms")
            print(f"  Decode: {pd['decode_ms_per_token']:.3f} ms/token")
            print(f"  Ratio: {pd['prefill_decode_ratio']:.2f}")
        
        # Throughput
        tp = results['throughput']
        print(f"\nThroughput:")
        print(f"  Average: {tp['throughput']:.2f} ± {tp.get('throughput_std', 0):.2f} tokens/s")
        print(f"  Total: {tp['total_tokens']} tokens in {tp['total_time']:.2f}s")
        
        # Speedup
        if results['speedup_vs_baseline']:
            print(f"\nSpeedup vs Baseline: {results['speedup_vs_baseline']:.2f}x")
    
    def compare_with_baseline(
        self,
        results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare results with baseline.
        
        Args:
            results: Current results
            baseline_results: Baseline results
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        # Latency speedup
        if baseline_results['latency']['ms_per_token'] > 0:
            comparison['latency_speedup'] = (
                baseline_results['latency']['ms_per_token'] / 
                results['latency']['ms_per_token']
            )
        
        # TTFT speedup
        if baseline_results['ttft']['ttft_ms'] > 0:
            comparison['ttft_speedup'] = (
                baseline_results['ttft']['ttft_ms'] / 
                results['ttft']['ttft_ms']
            )
        
        # Throughput improvement
        if baseline_results['throughput']['throughput'] > 0:
            comparison['throughput_improvement'] = (
                results['throughput']['throughput'] / 
                baseline_results['throughput']['throughput']
            )
        
        logger.info(f"\nComparison vs Baseline:")
        logger.info(f"  Latency speedup: {comparison.get('latency_speedup', 0):.2f}x")
        logger.info(f"  TTFT speedup: {comparison.get('ttft_speedup', 0):.2f}x")
        logger.info(f"  Throughput improvement: {comparison.get('throughput_improvement', 0):.2f}x")
        
        return comparison
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path,
        comparison: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Save results to JSON.
        
        Args:
            results: Results dictionary
            output_path: Output file path
            comparison: Optional comparison metrics
        """
        output_data = {
            'evaluation_type': 'time_performance',
            'model': str(self.model.model_path),
            'results': results
        }
        
        if comparison:
            output_data['comparison'] = comparison
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=float)
        
        logger.info(f"\nResults saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    """
    Example: Evaluate time performance of a model
    
    Usage:
        from model_interface import HuggingFaceModel
        
        model = HuggingFaceModel("mistralai/Mistral-7B-v0.1")
        model.load()
        
        evaluator = TimePerformanceEvaluator(model, verbose=True)
        results = evaluator.run(
            num_warmup=3,
            num_runs=10,
            max_new_tokens=128,
            measure_prefill_decode_split=True
        )
        
        evaluator.save_results(
            results,
            Path("results/time_performance.json")
        )
    """
    pass