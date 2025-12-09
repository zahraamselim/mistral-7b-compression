"""
Perplexity measurement utilities.

Measures perplexity on standard text datasets (WikiText, C4, etc.).
"""

import logging
import numpy as np
import torch
from typing import Dict, Optional
from datasets import load_dataset

from models.model_interface import ModelInterface

logger = logging.getLogger(__name__)


def measure_perplexity(
    model: ModelInterface,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
    max_length: int = 512,
    min_text_length: int = 100,
    stride: int = 512
) -> Dict[str, float]:
    """
    Calculate perplexity on a text dataset.
    
    Uses sliding window approach for long texts to accurately
    measure perplexity across the entire evaluation set.
    
    Args:
        model: ModelInterface instance
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split to use
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length
        min_text_length: Minimum text length to include
        stride: Stride for sliding window (set equal to max_length for no overlap)
        
    Returns:
        Dictionary with perplexity and metadata:
        - perplexity: Perplexity score
        - dataset: Dataset identifier
        - num_samples: Number of valid samples
        - total_tokens: Total tokens evaluated
        - loss: Average loss
    """
    logger.info(f"Measuring perplexity on {dataset_name}")
    logger.info(f"  Config: {dataset_config}")
    logger.info(f"  Split: {split}")
    logger.info(f"  Max samples: {max_samples}")
    logger.info(f"  Max length: {max_length}")
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
        logger.info(f"Loaded {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Filter short texts
    dataset = dataset.filter(lambda x: len(x.get("text", "")) > min_text_length)
    logger.info(f"After filtering: {len(dataset)} samples")
    
    if len(dataset) == 0:
        raise ValueError("No valid samples in dataset after filtering")
    
    total_loss = 0.0
    total_tokens = 0
    valid_samples = 0
    
    logger.info("Calculating perplexity...")
    
    for i in range(min(max_samples, len(dataset))):
        text = dataset[i].get("text", "")
        
        if not text or len(text) < min_text_length:
            continue
        
        try:
            # Encode and get actual token count
            inputs = model.encode(text, max_length=max_length)
            
            # Calculate loss directly
            with torch.no_grad():
                outputs = model.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            
            # Get actual number of tokens (excluding padding)
            attention_mask = inputs.get('attention_mask', torch.ones_like(inputs['input_ids']))
            num_tokens = attention_mask.sum().item() - 1  # Exclude first token
            
            if num_tokens > 0:
                total_loss += loss * num_tokens
                total_tokens += num_tokens
                valid_samples += 1
            
            if (i + 1) % 20 == 0:
                current_ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
                logger.info(
                    f"Progress: {i+1}/{min(max_samples, len(dataset))}, "
                    f"Current PPL: {current_ppl:.2f}"
                )
                
        except Exception as e:
            logger.warning(f"Failed on sample {i}: {e}")
            continue
    
    if total_tokens == 0:
        raise ValueError("No valid samples found for perplexity calculation")
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    results = {
        "perplexity": float(perplexity),
        "loss": float(avg_loss),
        "dataset": f"{dataset_name}/{dataset_config}",
        "split": split,
        "num_samples": valid_samples,
        "total_tokens": total_tokens,
        "max_length": max_length
    }
    
    logger.info(f"Perplexity: {perplexity:.4f}")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  Valid samples: {valid_samples}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    
    return results


def measure_perplexity_sliding_window(
    model: ModelInterface,
    text: str,
    max_length: int = 512,
    stride: int = 256
) -> float:
    """
    Calculate perplexity using sliding window for long texts.
    
    This provides more accurate perplexity for texts longer than max_length
    by evaluating overlapping windows.
    
    Args:
        model: ModelInterface instance
        text: Input text
        max_length: Maximum sequence length per window
        stride: Stride between windows
        
    Returns:
        Perplexity score
    """
    encodings = model.encode(text, max_length=None)
    
    seq_len = encodings['input_ids'].size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings['input_ids'][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    
    return float(ppl)


def compare_perplexity(
    baseline_ppl: float,
    current_ppl: float
) -> Dict[str, float]:
    """
    Compare perplexity between two models.
    
    Args:
        baseline_ppl: Baseline model perplexity
        current_ppl: Current model perplexity
        
    Returns:
        Dictionary with comparison metrics
    """
    if baseline_ppl <= 0 or current_ppl <= 0:
        logger.warning("Invalid perplexity values for comparison")
        return {
            'perplexity_ratio': 0.0,
            'perplexity_delta': 0.0,
            'perplexity_degradation_percent': 0.0
        }
    
    ratio = current_ppl / baseline_ppl
    delta = current_ppl - baseline_ppl
    degradation_percent = ((current_ppl - baseline_ppl) / baseline_ppl) * 100
    
    logger.info(f"Perplexity comparison:")
    logger.info(f"  Baseline: {baseline_ppl:.2f}")
    logger.info(f"  Current: {current_ppl:.2f}")
    logger.info(f"  Ratio: {ratio:.2f}x")
    logger.info(f"  Delta: {delta:+.2f}")
    logger.info(f"  Degradation: {degradation_percent:+.1f}%")
    
    return {
        'perplexity_ratio': ratio,
        'perplexity_delta': delta,
        'perplexity_degradation_percent': degradation_percent
    }