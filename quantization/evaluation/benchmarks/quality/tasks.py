"""Task evaluation using lm-evaluation-harness."""

import logging
from typing import List, Dict, Optional, Union, Any

logger = logging.getLogger(__name__)

TASK_REGISTRY = {
    'hellaswag': {'description': 'Sentence completion', 'metric': 'acc_norm', 'default_fewshot': 0, 'category': 'commonsense'},
    'winogrande': {'description': 'Pronoun resolution', 'metric': 'acc', 'default_fewshot': 0, 'category': 'commonsense'},
    'piqa': {'description': 'Physical interactions', 'metric': 'acc_norm', 'default_fewshot': 0, 'category': 'commonsense'},
    'siqa': {'description': 'Social interactions', 'metric': 'acc', 'default_fewshot': 0, 'category': 'commonsense'},
    'openbookqa': {'description': 'Elementary science', 'metric': 'acc_norm', 'default_fewshot': 0, 'category': 'commonsense'},
    'arc_easy': {'description': 'Science questions (easy)', 'metric': 'acc_norm', 'default_fewshot': 0, 'category': 'commonsense'},
    'arc_challenge': {'description': 'Science questions (hard)', 'metric': 'acc_norm', 'default_fewshot': 0, 'category': 'commonsense'},
    'commonsense_qa': {'description': 'General knowledge', 'metric': 'acc', 'default_fewshot': 0, 'category': 'commonsense'},
    'nq_open': {'description': 'Natural Questions', 'metric': 'exact_match', 'default_fewshot': 5, 'category': 'knowledge'},
    'triviaqa': {'description': 'Trivia questions', 'metric': 'exact_match', 'default_fewshot': 5, 'category': 'knowledge'},
    'boolq': {'description': 'Boolean questions', 'metric': 'acc', 'default_fewshot': 0, 'category': 'reading'},
    'quac': {'description': 'Conversational QA', 'metric': 'f1', 'default_fewshot': 0, 'category': 'reading'},
    'gsm8k': {'description': 'Grade school math', 'metric': 'exact_match', 'default_fewshot': 8, 'category': 'math'},
    'hendrycks_math': {'description': 'Competition math', 'metric': 'exact_match', 'default_fewshot': 4, 'category': 'math'},
    'math_algebra': {'description': 'MATH Algebra subset', 'metric': 'exact_match', 'default_fewshot': 4, 'category': 'math'},
    'humaneval': {'description': 'Python code (pass@1)', 'metric': 'pass@1', 'default_fewshot': 0, 'category': 'code'},
    'mbpp': {'description': 'Python problems', 'metric': 'pass@1', 'default_fewshot': 3, 'category': 'code'},
    'mmlu': {'description': 'Multitask understanding', 'metric': 'acc', 'default_fewshot': 5, 'category': 'aggregate'},
    'bbh': {'description': 'BIG-Bench Hard', 'metric': 'acc', 'default_fewshot': 3, 'category': 'aggregate'},
    'agieval': {'description': 'AGI Eval (English)', 'metric': 'acc', 'default_fewshot': 3, 'category': 'aggregate'},
    'lambada': {'description': 'Word prediction', 'metric': 'acc', 'default_fewshot': 0, 'category': 'language'},
    'storycloze': {'description': 'Story completion', 'metric': 'acc', 'default_fewshot': 0, 'category': 'language'},
}


def parse_task_config(task_config: Union[bool, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Parse task configuration."""
    if isinstance(task_config, bool):
        return {'enabled': task_config} if task_config else None
    elif isinstance(task_config, dict):
        if not task_config.get('enabled', True):
            return None
        return task_config
    return None


def get_metric_from_results(task_results: Dict[str, Any], task_name: str) -> Optional[float]:
    """Extract metric from task results."""
    task_info = TASK_REGISTRY.get(task_name, {})
    preferred_metric = task_info.get('metric', 'acc')
    
    metric_variations = [
        preferred_metric,
        f"{preferred_metric},none",
        f"{preferred_metric}_norm",
        f"{preferred_metric}_norm,none",
        "acc_norm",
        "acc_norm,none",
        "acc",
        "acc,none",
        "exact_match",
        "exact_match,none",
        "pass@1",
        "f1",
        "em"
    ]
    
    for metric_name in metric_variations:
        if metric_name in task_results:
            value = task_results[metric_name]
            
            if isinstance(value, dict):
                for key in ['mean', 'value', 'score']:
                    if key in value:
                        return float(value[key])
            elif isinstance(value, (int, float)):
                return float(value)
    
    for key, value in task_results.items():
        if isinstance(value, (int, float)):
            logger.debug(f"Using fallback metric '{key}' for {task_name}")
            return float(value)
        elif isinstance(value, dict):
            for subkey in ['mean', 'value', 'score']:
                if subkey in value and isinstance(value[subkey], (int, float)):
                    logger.debug(f"Using fallback metric '{key}.{subkey}' for {task_name}")
                    return float(value[subkey])
    
    logger.warning(f"No valid metric found for {task_name}. Available: {list(task_results.keys())}")
    return None


def run_lm_eval_tasks(
    model_interface,
    tasks: Union[List[str], Dict[str, Any]],
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Run tasks using lm-evaluation-harness.
    
    Args:
        model_interface: ModelInterface instance
        tasks: List of task names or dict with configs
        num_fewshot: Default number of few-shot examples
        limit: Limit number of samples per task
        batch_size: Batch size
        
    Returns:
        Dictionary mapping task names to scores
    """
    try:
        try:
            from lm_eval import simple_evaluate
            use_new_api = True
            logger.debug("Using lm-eval API v0.4.0+")
        except ImportError:
            try:
                from lm_eval import evaluator
                use_new_api = False
                logger.debug("Using lm-eval API pre-v0.4.0")
            except ImportError:
                logger.error("lm-eval not installed. Install: pip install lm-eval")
                return {}
        
        task_list = []
        task_configs = {}
        
        if isinstance(tasks, dict):
            for task_name, task_config in tasks.items():
                parsed_config = parse_task_config(task_config)
                if parsed_config is not None:
                    task_list.append(task_name)
                    task_configs[task_name] = parsed_config
        elif isinstance(tasks, list):
            task_list = tasks
            task_configs = {task: {} for task in tasks}
        else:
            logger.error(f"Invalid tasks format: {type(tasks)}")
            return {}
        
        if not task_list:
            logger.warning("No tasks enabled")
            return {}
        
        logger.info(f"Running {len(task_list)} tasks")
        
        lm_eval_model = model_interface.get_lm_eval_model()
        
        if lm_eval_model is None:
            logger.error("Model does not support lm-eval")
            return {}
        
        all_metrics = {}
        
        for task_name in task_list:
            try:
                task_cfg = task_configs.get(task_name, {})
                
                if 'num_fewshot' in task_cfg:
                    task_fewshot = task_cfg['num_fewshot']
                elif num_fewshot is not None:
                    task_fewshot = num_fewshot
                else:
                    task_info = TASK_REGISTRY.get(task_name, {})
                    task_fewshot = task_info.get('default_fewshot', 0)
                
                task_limit = task_cfg.get('limit', limit)
                task_batch = task_cfg.get('batch_size', batch_size)
                
                logger.info(f"Evaluating {task_name} ({task_fewshot}-shot)")
                
                if use_new_api:
                    results = simple_evaluate(
                        model=lm_eval_model,
                        tasks=[task_name],
                        num_fewshot=task_fewshot,
                        limit=task_limit,
                        batch_size=task_batch,
                        log_samples=False,
                        gen_kwargs=None,
                        task_manager=None,
                        verbosity="INFO",
                        predict_only=False,
                        random_seed=0,
                        numpy_random_seed=1234,
                        torch_random_seed=1234,
                        fewshot_random_seed=1234,
                        write_out=False,
                        apply_chat_template=False,
                        check_integrity=False,
                        confirm_run_unsafe_code=True
                    )
                else:
                    results = evaluator.simple_evaluate(
                        model=lm_eval_model,
                        tasks=[task_name],
                        num_fewshot=task_fewshot,
                        limit=task_limit,
                        batch_size=task_batch
                    )
                
                if 'results' in results and task_name in results['results']:
                    task_results = results['results'][task_name]
                    score = get_metric_from_results(task_results, task_name)
                    
                    if score is not None:
                        all_metrics[task_name] = score
                        logger.info(f"  {task_name}: {score:.4f} ({score*100:.2f}%)")
                    else:
                        logger.warning(f"  {task_name}: No valid metric found")
                else:
                    logger.warning(f"  {task_name}: No results returned")
                
            except Exception as e:
                logger.error(f"  {task_name} failed: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    import traceback
                    traceback.print_exc()
                continue
        
        if not all_metrics:
            logger.warning("No metrics extracted")
        else:
            logger.info(f"Successfully evaluated {len(all_metrics)}/{len(task_list)} tasks")
        
        return all_metrics
        
    except ImportError as e:
        logger.error(f"lm-eval not installed: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error running lm-eval: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return {}


def list_available_tasks() -> Dict[str, Dict[str, Any]]:
    """Get list of all supported tasks."""
    return TASK_REGISTRY.copy()


def get_task_info(task_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific task."""
    return TASK_REGISTRY.get(task_name)


def get_tasks_by_category(category: str) -> List[str]:
    """Get all tasks in a category."""
    return [
        task_name for task_name, info in TASK_REGISTRY.items()
        if info.get('category') == category
    ]