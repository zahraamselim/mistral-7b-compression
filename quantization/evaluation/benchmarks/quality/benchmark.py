"""Quality benchmark for quantized models."""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from core.benchmark import Benchmark
from core.result import BenchmarkResult
from benchmarks.quality.perplexity import PerplexityEvaluator
from benchmarks.quality.tasks import run_lm_eval_tasks

logger = logging.getLogger(__name__)


@dataclass
class QualityResult(BenchmarkResult):
    """Results from quality benchmarks."""
    
    perplexity: Optional[float] = None
    perplexity_dataset: Optional[str] = None
    task_scores: Dict[str, float] = field(default_factory=dict)
    average_score: Optional[float] = None
    num_tasks: int = 0
    
    def __str__(self) -> str:
        """Format results for display."""
        lines = []
        lines.append("")
        lines.append("Quality Benchmark Results")
        lines.append("")
        
        if self.perplexity is not None:
            dataset_info = f" on {self.perplexity_dataset}" if self.perplexity_dataset else ""
            lines.append(f"Perplexity{dataset_info}: {self.perplexity:.4f}")
            lines.append("")
        
        if self.task_scores:
            lines.append(f"Task Evaluation ({self.num_tasks} tasks)")
            lines.append("")
            
            categories = {
                'Commonsense Reasoning': [
                    'hellaswag', 'winogrande', 'piqa', 'siqa',
                    'openbookqa', 'arc_easy', 'arc_challenge', 'commonsense_qa'
                ],
                'World Knowledge': ['nq_open', 'triviaqa'],
                'Reading Comprehension': ['boolq', 'quac'],
                'Math': ['gsm8k', 'hendrycks_math', 'math_algebra'],
                'Code': ['humaneval', 'mbpp'],
                'Aggregate Benchmarks': ['mmlu', 'bbh', 'agieval'],
                'Language Understanding': ['lambada', 'storycloze'],
            }
            
            for category, task_list in categories.items():
                category_tasks = {k: v for k, v in self.task_scores.items() if k in task_list}
                if category_tasks:
                    lines.append(f"{category}:")
                    for task, score in sorted(category_tasks.items()):
                        lines.append(f"  {task:.<30} {score*100:>6.2f}%")
                    lines.append("")
            
            uncategorized = set(self.task_scores.keys()) - set(
                task for tasks in categories.values() for task in tasks
            )
            if uncategorized:
                lines.append("Other Tasks:")
                for task in sorted(uncategorized):
                    score = self.task_scores[task]
                    lines.append(f"  {task:.<30} {score*100:>6.2f}%")
                lines.append("")
            
            if self.average_score is not None:
                lines.append(f"Average Score: {self.average_score*100:.2f}%")
        
        return '\n'.join(lines)


class QualityBenchmark(Benchmark[QualityResult]):
    """
    Benchmark suite for measuring quantized model quality.
    
    Evaluates:
    - Perplexity on text datasets
    - Task accuracy via lm-eval-harness
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        verbose: bool = False
    ):
        """
        Initialize quality benchmark.
        
        Args:
            model_interface: ModelInterface instance
            config: Quality config
            verbose: Enable verbose logging
        """
        super().__init__(
            model_interface=model_interface,
            config=config,
            verbose=verbose
        )
        
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        self.perplexity_evaluator = PerplexityEvaluator(model_interface)
        
        logger.info("Quality benchmark initialized")
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        if not super().validate_config():
            return False
        
        perplexity_config = self.config.get('perplexity', {})
        tasks_config = self.config.get('tasks', {})
        
        has_perplexity = perplexity_config.get('enabled', False)
        has_tasks = tasks_config.get('enabled', False)
        
        if not has_perplexity and not has_tasks:
            logger.warning("No quality metrics enabled")
        
        return True
    
    def run(
        self,
        measure_perplexity: Optional[bool] = None,
        run_tasks: Optional[bool] = None,
        perplexity_kwargs: Optional[Dict[str, Any]] = None,
        tasks_kwargs: Optional[Dict[str, Any]] = None
    ) -> QualityResult:
        """
        Run quality benchmarks.
        
        Args:
            measure_perplexity: Whether to measure perplexity
            run_tasks: Whether to run task evaluation
            perplexity_kwargs: Additional perplexity kwargs
            tasks_kwargs: Additional task kwargs
            
        Returns:
            QualityResult with all metrics
        """
        self.validate_config()
        
        perplexity_config = self.config.get('perplexity', {})
        tasks_config = self.config.get('tasks', {})
        
        measure_perplexity = (
            measure_perplexity if measure_perplexity is not None
            else perplexity_config.get('enabled', True)
        )
        run_tasks = (
            run_tasks if run_tasks is not None
            else tasks_config.get('enabled', False)
        )
        
        logger.info("")
        logger.info("Starting quality benchmarks")
        logger.info("")
        
        result = QualityResult()
        
        if measure_perplexity:
            logger.info("Perplexity Evaluation")
            perplexity_result = self._measure_perplexity(
                perplexity_config,
                perplexity_kwargs
            )
            if perplexity_result:
                result.perplexity = perplexity_result['perplexity']
                result.perplexity_dataset = perplexity_result.get('dataset')
        
        if run_tasks:
            logger.info("")
            logger.info("Task Evaluation")
            task_scores = self._run_tasks(tasks_config, tasks_kwargs)
            result.task_scores = task_scores
            result.num_tasks = len(task_scores)
            
            if task_scores:
                result.average_score = sum(task_scores.values()) / len(task_scores)
        
        logger.info("")
        logger.info("Quality benchmarks complete")
        
        if self.verbose:
            print(result)
        
        return result
    
    def _measure_perplexity(
        self,
        config: Dict[str, Any],
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Measure perplexity on dataset."""
        try:
            kwargs = dict(config)
            if additional_kwargs:
                kwargs.update(additional_kwargs)
            
            kwargs.pop('enabled', None)
            
            dataset_name = kwargs.pop('dataset', 'wikitext')
            dataset_config = kwargs.pop('dataset_config', 'wikitext-2-raw-v1')
            split = kwargs.pop('split', 'test')
            num_samples = kwargs.pop('num_samples', 100)
            max_length = kwargs.pop('max_length', 512)
            stride = kwargs.pop('stride', None)
            batch_size = kwargs.pop('batch_size', 1)
            
            logger.info(f"Dataset: {dataset_name}/{dataset_config}")
            logger.info(f"Samples: {num_samples}, Max length: {max_length}")
            
            perplexity = self.perplexity_evaluator.calculate(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                num_samples=num_samples,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size
            )
            
            logger.info(f"Perplexity: {perplexity:.4f}")
            
            return {
                'perplexity': perplexity,
                'dataset': f"{dataset_name}/{dataset_config}"
            }
            
        except Exception as e:
            logger.error(f"Perplexity measurement failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _run_tasks(
        self,
        config: Dict[str, Any],
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Run task evaluation."""
        try:
            if not self.model_interface.supports_lm_eval():
                logger.warning("Model does not support lm-eval")
                return {}
            
            task_configs = config.get('task_list', {})
            
            enabled_tasks = {}
            for task_name, task_cfg in task_configs.items():
                if isinstance(task_cfg, dict) and task_cfg.get('enabled', False):
                    enabled_tasks[task_name] = task_cfg
            
            if not enabled_tasks:
                logger.info("No tasks enabled")
                return {}
            
            logger.info(f"Running {len(enabled_tasks)} tasks")
            
            batch_size = config.get('batch_size', 1)
            
            scores = run_lm_eval_tasks(
                model_interface=self.model_interface,
                tasks=enabled_tasks,
                num_fewshot=None,
                limit=None,
                batch_size=batch_size
            )
            
            if scores:
                logger.info("")
                logger.info("Results:")
                for task, score in sorted(scores.items()):
                    logger.info(f"  {task:.<30} {score*100:>6.2f}%")
                
                avg_score = sum(scores.values()) / len(scores)
                logger.info(f"  Average:                       {avg_score*100:>6.2f}%")
            
            return scores
            
        except Exception as e:
            logger.error(f"Task evaluation failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {}