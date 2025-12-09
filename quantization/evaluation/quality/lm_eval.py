"""
LM Evaluation Harness integration.

Provides convenient wrappers for running lm-evaluation-harness tasks.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def run_lm_eval_tasks(
    model_path: str,
    tasks: List[str],
    output_dir: Path,
    num_fewshot: int = 0,
    batch_size: int = 1,
    device: str = "cuda:0",
    model_args: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run lm-evaluation-harness tasks.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        tasks: List of task names (e.g., ["hellaswag", "arc_easy"])
        output_dir: Directory to save results
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        device: Device to use
        model_args: Additional model arguments (e.g., "trust_remote_code=True")
        limit: Limit number of examples per task (for testing)
        
    Returns:
        Dictionary mapping task names to results
    """
    logger.info("Running lm-evaluation-harness tasks")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Tasks: {tasks}")
    logger.info(f"  Few-shot: {num_fewshot}")
    logger.info(f"  Batch size: {batch_size}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for task in tasks:
        logger.info(f"\nTask: {task}")
        
        output_file = output_dir / f"lm_eval_{task}.json"
        
        if model_args is None:
            model_args_str = f"pretrained={model_path},torch_dtype=float16"
        else:
            model_args_str = f"pretrained={model_path},{model_args}"
        
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", model_args_str,
            "--tasks", task,
            "--num_fewshot", str(num_fewshot),
            "--device", device,
            "--batch_size", str(batch_size),
            "--output_path", str(output_file)
        ]
        
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("Task completed successfully")
            
            task_results = parse_lm_eval_results(output_file)
            all_results[task] = task_results
            
            if task_results:
                logger.info(f"Results: {task_results}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Task {task} failed with return code {e.returncode}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            all_results[task] = {"error": str(e)}
        except Exception as e:
            logger.error(f"Task {task} failed: {e}")
            all_results[task] = {"error": str(e)}
    
    return all_results


def parse_lm_eval_results(output_file: Path) -> Dict[str, float]:
    """
    Parse lm-evaluation-harness output file.
    
    Args:
        output_file: Path to lm_eval output JSON file
        
    Returns:
        Dictionary with parsed metrics
    """
    if not output_file.exists():
        logger.warning(f"Output file not found: {output_file}")
        return {}
    
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        results = {}
        
        if "results" in data:
            for task_name, task_data in data["results"].items():
                if isinstance(task_data, dict):
                    for metric_name, metric_value in task_data.items():
                        if isinstance(metric_value, (int, float)):
                            results[f"{task_name}_{metric_name}"] = metric_value
                        elif isinstance(metric_value, dict) and "value" in metric_value:
                            results[f"{task_name}_{metric_name}"] = metric_value["value"]
        
        logger.debug(f"Parsed results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to parse lm_eval results: {e}")
        return {}


def run_reasoning_suite(
    model_path: str,
    output_dir: Path,
    num_fewshot: int = 0,
    batch_size: int = 1,
    device: str = "cuda:0",
    limit: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run standard reasoning task suite.
    
    Tasks:
    - HellaSwag: Common sense reasoning
    - WinoGrande: Pronoun resolution
    - PIQA: Physical reasoning
    - ARC-Easy/Challenge: Science questions
    - OpenBookQA: Science questions with retrieval
    - BoolQ: Yes/no questions
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Directory to save results
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples per task
        
    Returns:
        Dictionary with reasoning task results
    """
    tasks = [
        "hellaswag",
        "winogrande",
        "piqa",
        "arc_easy",
        "arc_challenge",
        "openbookqa",
        "boolq"
    ]
    
    logger.info("Running reasoning task suite")
    
    return run_lm_eval_tasks(
        model_path=model_path,
        tasks=tasks,
        output_dir=output_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit
    )


def run_rag_suite(
    model_path: str,
    output_dir: Path,
    num_fewshot: int = 0,
    batch_size: int = 1,
    device: str = "cuda:0",
    limit: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run RAG-focused task suite from lm-evaluation-harness.
    
    These tasks test reading comprehension, retrieval, and context understanding
    which are critical for RAG applications.
    
    Tasks:
    - SQuAD: Reading comprehension from Wikipedia passages
    - NaturalQuestions: Open-domain QA requiring retrieval
    - TriviaQA: Question answering with evidence documents
    - DROP: Discrete reasoning over paragraphs (math + reading)
    - QuAC: Question answering in conversational context
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Directory to save results
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples per task
        
    Returns:
        Dictionary with RAG task results
    """
    tasks = [
        "squad",
        "nq_open",
        "triviaqa",
        "drop",
        "quac"
    ]
    
    logger.info("Running RAG-focused task suite")
    logger.info("These tasks test reading comprehension and context understanding")
    
    return run_lm_eval_tasks(
        model_path=model_path,
        tasks=tasks,
        output_dir=output_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit
    )


def run_mmlu(
    model_path: str,
    output_dir: Path,
    num_fewshot: int = 5,
    batch_size: int = 1,
    device: str = "cuda:0",
    limit: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run MMLU (Massive Multitask Language Understanding) benchmark.
    
    MMLU tests knowledge across 57 subjects including STEM, humanities,
    social sciences, and more.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Directory to save results
        num_fewshot: Number of few-shot examples (standard is 5)
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples per task
        
    Returns:
        Dictionary with MMLU results
    """
    logger.info("Running MMLU benchmark")
    logger.info(f"  Few-shot: {num_fewshot} (standard)")
    
    return run_lm_eval_tasks(
        model_path=model_path,
        tasks=["mmlu"],
        output_dir=output_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit
    )


def run_truthfulqa(
    model_path: str,
    output_dir: Path,
    batch_size: int = 1,
    device: str = "cuda:0",
    limit: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run TruthfulQA benchmark.
    
    Tests model's ability to generate truthful answers and avoid
    common misconceptions.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples
        
    Returns:
        Dictionary with TruthfulQA results
    """
    logger.info("Running TruthfulQA benchmark")
    
    return run_lm_eval_tasks(
        model_path=model_path,
        tasks=["truthfulqa_mc"],
        output_dir=output_dir,
        num_fewshot=0,
        batch_size=batch_size,
        device=device,
        limit=limit
    )


def run_gsm8k(
    model_path: str,
    output_dir: Path,
    num_fewshot: int = 5,
    batch_size: int = 1,
    device: str = "cuda:0",
    limit: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run GSM8K (Grade School Math) benchmark.
    
    Tests mathematical reasoning with grade school level problems.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        output_dir: Directory to save results
        num_fewshot: Number of few-shot examples
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples
        
    Returns:
        Dictionary with GSM8K results
    """
    logger.info("Running GSM8K benchmark")
    
    return run_lm_eval_tasks(
        model_path=model_path,
        tasks=["gsm8k"],
        output_dir=output_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit
    )