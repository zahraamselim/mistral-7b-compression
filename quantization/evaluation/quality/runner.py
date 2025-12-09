"""
Quality benchmark orchestrator.

Coordinates perplexity measurement and lm-evaluation-harness tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from models.model_interface import ModelInterface
from quality.perplexity import measure_perplexity, compare_perplexity
from quality.lm_eval import (
    run_reasoning_suite,
    run_mmlu,
    run_truthfulqa,
    run_gsm8k,
    run_lm_eval_tasks
)

logger = logging.getLogger(__name__)


@dataclass
class QualityResults:
    """Results from quality benchmarks."""
    
    # Perplexity
    perplexity: float = 0.0
    perplexity_dataset: Optional[str] = None
    perplexity_loss: Optional[float] = None
    perplexity_num_samples: Optional[int] = None
    perplexity_total_tokens: Optional[int] = None
    
    # Reasoning tasks (if run)
    hellaswag_acc: Optional[float] = None
    hellaswag_acc_norm: Optional[float] = None
    winogrande_acc: Optional[float] = None
    piqa_acc: Optional[float] = None
    arc_easy_acc: Optional[float] = None
    arc_challenge_acc: Optional[float] = None
    openbookqa_acc: Optional[float] = None
    boolq_acc: Optional[float] = None
    
    # MMLU (if run)
    mmlu_acc: Optional[float] = None
    
    # Additional benchmarks
    truthfulqa_acc: Optional[float] = None
    gsm8k_acc: Optional[float] = None
    
    # Comparison metrics (vs baseline)
    perplexity_ratio: Optional[float] = None
    perplexity_degradation_percent: Optional[float] = None
    
    # Raw results
    raw_lm_eval_results: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return asdict(self)


class QualityBenchmark:
    """
    Comprehensive quality benchmark suite.
    
    Measures:
    - Perplexity on WikiText or custom dataset
    - Standard reasoning tasks (via lm-evaluation-harness)
    - MMLU knowledge benchmark
    - Optional: TruthfulQA, GSM8K, and other tasks
    """
    
    def __init__(
        self,
        model: ModelInterface,
        model_path: str,
        verbose: bool = False
    ):
        """
        Initialize quality benchmark.
        
        Args:
            model: ModelInterface instance (for perplexity)
            model_path: Path to model for lm-eval (can be HF model ID or local path)
            verbose: Enable verbose logging
        """
        self.model = model
        self.model_path = model_path
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        logger.info("Quality Benchmark initialized")
        logger.info(f"  Model path: {model_path}")
    
    def run_perplexity(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "test",
        max_samples: int = 100,
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Measure perplexity on a text dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split
            max_samples: Maximum samples to evaluate
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with perplexity metrics
        """
        logger.info("Running Perplexity Measurement")
        
        results = measure_perplexity(
            self.model,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            max_samples=max_samples,
            max_length=max_length
        )
        
        return results
    
    def run_reasoning_tasks(
        self,
        output_dir: Path,
        num_fewshot: int = 0,
        batch_size: int = 1,
        device: str = "cuda:0",
        limit: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run standard reasoning task suite.
        
        Args:
            output_dir: Directory to save results
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            device: Device to use
            limit: Limit examples per task (for testing)
            
        Returns:
            Dictionary with reasoning task results
        """
        logger.info("Running Reasoning Tasks")
        
        return run_reasoning_suite(
            self.model_path,
            output_dir,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit
        )
    
    def run_mmlu_benchmark(
        self,
        output_dir: Path,
        num_fewshot: int = 5,
        batch_size: int = 1,
        device: str = "cuda:0",
        limit: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run MMLU benchmark.
        
        Args:
            output_dir: Directory to save results
            num_fewshot: Number of few-shot examples (standard is 5)
            batch_size: Batch size for evaluation
            device: Device to use
            limit: Limit examples (for testing)
            
        Returns:
            Dictionary with MMLU results
        """
        logger.info("Running MMLU Benchmark")
        
        return run_mmlu(
            self.model_path,
            output_dir,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit
        )
    
    def run_all(
        self,
        output_dir: Path,
        perplexity_dataset: str = "wikitext",
        perplexity_config: str = "wikitext-2-raw-v1",
        perplexity_samples: int = 100,
        include_reasoning: bool = True,
        include_mmlu: bool = False,
        include_truthfulqa: bool = False,
        include_gsm8k: bool = False,
        num_fewshot: int = 0,
        batch_size: int = 1,
        device: str = "cuda:0",
        limit: Optional[int] = None,
        baseline_results: Optional[Dict[str, Any]] = None
    ) -> QualityResults:
        """
        Run all quality benchmarks.
        
        Args:
            output_dir: Directory to save results
            perplexity_dataset: Dataset for perplexity
            perplexity_config: Dataset configuration
            perplexity_samples: Number of samples for perplexity
            include_reasoning: Whether to run reasoning tasks
            include_mmlu: Whether to run MMLU
            include_truthfulqa: Whether to run TruthfulQA
            include_gsm8k: Whether to run GSM8K
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            device: Device to use
            limit: Limit examples per task (for testing)
            baseline_results: Baseline results for comparison
            
        Returns:
            QualityResults object with all metrics
        """
        logger.info("QUALITY BENCHMARK SUITE")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Output: {output_dir}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = QualityResults()
        
        # 1. Perplexity
        logger.info("\n[1] Perplexity Measurement")
        perplexity_results = self.run_perplexity(
            dataset_name=perplexity_dataset,
            dataset_config=perplexity_config,
            max_samples=perplexity_samples
        )
        
        results.perplexity = perplexity_results['perplexity']
        results.perplexity_dataset = perplexity_results['dataset']
        results.perplexity_loss = perplexity_results.get('loss')
        results.perplexity_num_samples = perplexity_results.get('num_samples')
        results.perplexity_total_tokens = perplexity_results.get('total_tokens')
        
        self._save_json(perplexity_results, output_dir / "perplexity.json")
        
        # Compare with baseline if provided
        if baseline_results and 'perplexity' in baseline_results:
            comparison = compare_perplexity(
                baseline_results['perplexity'],
                results.perplexity
            )
            results.perplexity_ratio = comparison['perplexity_ratio']
            results.perplexity_degradation_percent = comparison['perplexity_degradation_percent']
        
        # 2. Reasoning tasks
        if include_reasoning:
            logger.info("\n[2] Reasoning Tasks")
            reasoning_results = self.run_reasoning_tasks(
                output_dir,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=limit
            )
            
            # Extract metrics
            results.raw_lm_eval_results = reasoning_results
            self._extract_reasoning_metrics(results, reasoning_results)
            self._save_json(reasoning_results, output_dir / "reasoning_tasks.json")
        
        # 3. MMLU
        if include_mmlu:
            logger.info("\n[3] MMLU Benchmark")
            mmlu_results = run_mmlu(
                self.model_path,
                output_dir,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=limit
            )
            
            results.mmlu_acc = self._extract_metric(mmlu_results, 'mmlu', 'acc')
            self._save_json(mmlu_results, output_dir / "mmlu.json")
        
        # 4. TruthfulQA
        if include_truthfulqa:
            logger.info("\n[4] TruthfulQA")
            truthfulqa_results = run_truthfulqa(
                self.model_path,
                output_dir,
                batch_size=batch_size,
                device=device,
                limit=limit
            )
            
            results.truthfulqa_acc = self._extract_metric(truthfulqa_results, 'truthfulqa', 'acc')
            self._save_json(truthfulqa_results, output_dir / "truthfulqa.json")
        
        # 5. GSM8K
        if include_gsm8k:
            logger.info("\n[5] GSM8K")
            gsm8k_results = run_gsm8k(
                self.model_path,
                output_dir,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=limit
            )
            
            results.gsm8k_acc = self._extract_metric(gsm8k_results, 'gsm8k', 'acc')
            self._save_json(gsm8k_results, output_dir / "gsm8k.json")
        
        # Save complete results
        self._save_json(results.to_dict(), output_dir / "quality_complete_results.json")
        
        logger.info("QUALITY BENCHMARKS COMPLETE")
        
        if self.verbose:
            self._print_summary(results)
        
        return results
    
    def _extract_reasoning_metrics(
        self,
        results: QualityResults,
        reasoning_results: Dict[str, Dict[str, float]]
    ) -> None:
        """Extract reasoning task metrics into QualityResults."""
        results.hellaswag_acc = self._extract_metric(reasoning_results, 'hellaswag', 'acc')
        results.hellaswag_acc_norm = self._extract_metric(reasoning_results, 'hellaswag', 'acc_norm')
        results.winogrande_acc = self._extract_metric(reasoning_results, 'winogrande', 'acc')
        results.piqa_acc = self._extract_metric(reasoning_results, 'piqa', 'acc')
        results.arc_easy_acc = self._extract_metric(reasoning_results, 'arc_easy', 'acc')
        results.arc_challenge_acc = self._extract_metric(reasoning_results, 'arc_challenge', 'acc')
        results.openbookqa_acc = self._extract_metric(reasoning_results, 'openbookqa', 'acc')
        results.boolq_acc = self._extract_metric(reasoning_results, 'boolq', 'acc')
    
    def _extract_metric(
        self,
        results: Dict[str, Dict[str, float]],
        task: str,
        metric: str
    ) -> Optional[float]:
        """Extract a specific metric from lm_eval results."""
        if task not in results:
            return None
        
        task_results = results[task]
        
        # Try different key formats
        for key in [f"{task}_{metric}", metric, f"{metric}"]:
            if key in task_results:
                return task_results[key]
        
        return None
    
    def _save_json(self, data: Dict, filepath: Path) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=float)
        
        logger.debug(f"Saved results to {filepath}")
    
    def _print_summary(self, results: QualityResults) -> None:
        """Print summary of results."""
        print("QUALITY RESULTS SUMMARY")
        
        print(f"\nPerplexity:")
        print(f"  {results.perplexity:.2f} on {results.perplexity_dataset}")
        if results.perplexity_degradation_percent:
            print(f"  Degradation: {results.perplexity_degradation_percent:+.1f}% vs baseline")
        
        if results.hellaswag_acc:
            print(f"\nReasoning Tasks:")
            if results.hellaswag_acc:
                print(f"  HellaSwag: {results.hellaswag_acc*100:.1f}%")
            if results.winogrande_acc:
                print(f"  WinoGrande: {results.winogrande_acc*100:.1f}%")
            if results.piqa_acc:
                print(f"  PIQA: {results.piqa_acc*100:.1f}%")
            if results.arc_easy_acc:
                print(f"  ARC-Easy: {results.arc_easy_acc*100:.1f}%")
            if results.arc_challenge_acc:
                print(f"  ARC-Challenge: {results.arc_challenge_acc*100:.1f}%")
        
        if results.mmlu_acc:
            print(f"\nMMLU: {results.mmlu_acc*100:.1f}%")
        
        if results.truthfulqa_acc:
            print(f"\nTruthfulQA: {results.truthfulqa_acc*100:.1f}%")
        
        if results.gsm8k_acc:
            print(f"\nGSM8K: {results.gsm8k_acc*100:.1f}%")
        