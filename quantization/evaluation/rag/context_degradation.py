"""
Context Degradation Benchmark

Measures accuracy degradation as context length increases in quantized models.
"""

import random
import numpy as np
import torch
import gc
import time
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from scipy import stats
import re

from model_interface import ModelInterface, GenerationConfig


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


class ContextDegradationBenchmark:
    """
    Measures accuracy degradation with increasing context length.
    
    Metrics:
        - Accuracy by context length (EM and F1)
        - Accuracy by answer position (start/middle/end)
        - Degradation slope (linear regression)
        - Cliff point (statistically significant drop)
        - R-squared for linear fit
        - Effect sizes (Cohen's d)
    """
    
    def __init__(
        self,
        model: ModelInterface,
        context_lengths: List[int] = None,
        samples_per_length: int = 100,
        answer_positions: List[str] = None,
        context_tolerance: float = 0.05,
        random_seed: int = 42
    ):
        self.model = model
        self.context_lengths = sorted(context_lengths or [512, 1024, 2048, 4096])
        self.samples_per_length = samples_per_length
        self.answer_positions = answer_positions or ["start", "middle", "end"]
        self.context_tolerance = context_tolerance
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.tokenized_chunks = None
    
    def _load_datasets(self) -> Tuple[any, List[Dict[str, any]]]:
        """Load datasets and pre-tokenize for efficiency."""
        print("Loading Natural Questions...")
        nq_dataset = load_dataset("nq_open", split="validation")
        
        print("Loading and pre-tokenizing Wikipedia...")
        try:
            wiki_dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True
            )
            
            tokenized_chunks = []
            for i, sample in enumerate(wiki_dataset):
                if i >= 1000:
                    break
                
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                
                sentences = re.split(r'[.!?]+\s+', text)
                
                for sent in sentences:
                    if len(sent.strip()) < 20:
                        continue
                    
                    try:
                        tokens = self.model.tokenizer.encode(
                            sent.strip() + ". ",
                            add_special_tokens=False
                        )
                        
                        if 10 <= len(tokens) <= 80:
                            tokenized_chunks.append({
                                'tokens': tokens,
                                'text': sent.strip() + ". ",
                                'length': len(tokens)
                            })
                    except Exception:
                        continue
                    
                    if len(tokenized_chunks) >= 5000:
                        break
                
                if len(tokenized_chunks) >= 5000:
                    break
            
        except Exception as e:
            print(f"Failed to load wikimedia/wikipedia: {e}")
            print("Falling back to simple-wikipedia...")
            
            wiki_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:1000]")
            tokenized_chunks = []
            
            for sample in wiki_dataset:
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                
                sentences = re.split(r'[.!?]+\s+', text)
                
                for sent in sentences:
                    if len(sent.strip()) < 20:
                        continue
                    
                    try:
                        tokens = self.model.tokenizer.encode(
                            sent.strip() + ". ",
                            add_special_tokens=False
                        )
                        
                        if 10 <= len(tokens) <= 80:
                            tokenized_chunks.append({
                                'tokens': tokens,
                                'text': sent.strip() + ". ",
                                'length': len(tokens)
                            })
                    except Exception:
                        continue
        
        print(f"Pre-tokenized {len(tokenized_chunks)} chunks")
        return nq_dataset, tokenized_chunks
    
    def _find_answer_passage(
        self,
        answer: str,
        tokenized_chunks: List[Dict[str, any]],
        min_context_length: int = 30
    ) -> Optional[Dict[str, any]]:
        """Find a passage that naturally contains the answer."""
        answer_lower = answer.lower()
        
        candidates = []
        for chunk in tokenized_chunks:
            if answer_lower in chunk['text'].lower():
                candidates.append(chunk)
                if len(candidates) >= 20:
                    break
        
        if not candidates:
            return None
        
        valid_candidates = [c for c in candidates if c['length'] >= min_context_length]
        if not valid_candidates:
            valid_candidates = candidates
        
        return random.choice(valid_candidates)
    
    def _get_position_fraction(self, position: str) -> Tuple[float, float]:
        """Convert position name to fraction range."""
        if position == "start":
            return (0.0, 0.15)
        elif position == "middle":
            return (0.4, 0.6)
        elif position == "end":
            return (0.85, 1.0)
        elif position == "random":
            return (0.2, 0.8)
        else:
            return (0.2, 0.8)
    
    def _build_context_from_tokens(
        self,
        answer_passage: Dict[str, any],
        target_length: int,
        answer_position: str,
        tokenized_chunks: List[Dict[str, any]]
    ) -> Tuple[List[int], int, int]:
        """Build context at exact token length with answer at specified position."""
        context_tokens = []
        
        pos_min, pos_max = self._get_position_fraction(answer_position)
        answer_position_frac = random.uniform(pos_min, pos_max)
        
        tokens_before = int(target_length * answer_position_frac)
        tokens_before = max(10, min(tokens_before, target_length - answer_passage['length'] - 10))
        
        current_length = 0
        filler_pool = [c for c in tokenized_chunks if c != answer_passage]
        
        while current_length < tokens_before - answer_passage['length']:
            chunk = random.choice(filler_pool)
            
            if current_length + chunk['length'] <= tokens_before:
                context_tokens.extend(chunk['tokens'])
                current_length += chunk['length']
            else:
                remaining = tokens_before - current_length
                if remaining > 5:
                    context_tokens.extend(chunk['tokens'][:remaining])
                    current_length += remaining
                break
        
        answer_start = len(context_tokens)
        context_tokens.extend(answer_passage['tokens'])
        answer_end = len(context_tokens)
        
        while len(context_tokens) < target_length:
            chunk = random.choice(filler_pool)
            
            if len(context_tokens) + chunk['length'] <= target_length:
                context_tokens.extend(chunk['tokens'])
            else:
                remaining = target_length - len(context_tokens)
                if remaining > 0:
                    context_tokens.extend(chunk['tokens'][:remaining])
                break
        
        return context_tokens, answer_start, answer_end
    
    def _test_context_length(
        self,
        context_length: int,
        answer_position: str,
        nq_dataset: any,
        tokenized_chunks: List[Dict[str, any]],
        config: GenerationConfig
    ) -> Dict[str, any]:
        """Test model at specific context length and answer position."""
        correct_em = 0
        f1_scores = []
        total = 0
        attempted = 0
        
        min_length = int(context_length * (1 - self.context_tolerance))
        max_length = int(context_length * (1 + self.context_tolerance))
        
        start_time = time.time()
        
        while total < self.samples_per_length and attempted < len(nq_dataset):
            sample = nq_dataset[attempted]
            attempted += 1
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not answers or not question:
                continue
            
            answer = answers[0]
            
            answer_passage = self._find_answer_passage(
                answer,
                tokenized_chunks,
                min_context_length=20
            )
            
            if not answer_passage:
                continue
            
            try:
                context_tokens, ans_start, ans_end = self._build_context_from_tokens(
                    answer_passage,
                    context_length,
                    answer_position,
                    tokenized_chunks
                )
                
                actual_length = len(context_tokens)
                if not (min_length <= actual_length <= max_length):
                    continue
                
                context_text = self.model.tokenizer.decode(
                    context_tokens,
                    skip_special_tokens=True
                )
                
                prompt = f"{context_text}\n\nQuestion: {question}\n\nAnswer:"
                
                output = self.model.generate(prompt, config)
                
                generated_lower = output.generated_text.lower()
                answer_lower = answer.lower()
                
                if answer_lower in generated_lower:
                    correct_em += 1
                
                f1 = max([compute_f1(output.generated_text, ans) for ans in answers])
                f1_scores.append(f1)
                
                total += 1
                
                if total % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / total
                    remaining = (self.samples_per_length - total) * avg_time
                    current_em = correct_em / total
                    current_f1 = np.mean(f1_scores)
                    
                    print(f"    [{total}/{self.samples_per_length}] "
                          f"EM={current_em:.3f} F1={current_f1:.3f} "
                          f"ETA={remaining/60:.1f}min")
                
                del output, context_tokens
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    continue
            except Exception:
                continue
        
        accuracy_em = correct_em / total if total > 0 else 0.0
        accuracy_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        
        return {
            "accuracy_em": float(accuracy_em),
            "accuracy_f1": accuracy_f1,
            "f1_std": float(np.std(f1_scores)) if f1_scores else 0.0,
            "correct": correct_em,
            "total": total,
            "attempted": attempted
        }
    
    def _calculate_cliff_point_statistical(
        self,
        lengths: List[int],
        accuracies: List[float],
        std_errors: List[float]
    ) -> Optional[Dict[str, any]]:
        """Identify context length with statistically significant performance drop."""
        if len(accuracies) < 2:
            return None
        
        baseline_acc = accuracies[0]
        baseline_se = std_errors[0]
        
        for i in range(1, len(accuracies)):
            current_acc = accuracies[i]
            current_se = std_errors[i]
            
            pooled_se = np.sqrt(baseline_se**2 + current_se**2)
            
            if pooled_se > 0:
                z_score = (baseline_acc - current_acc) / pooled_se
                p_value = stats.norm.sf(abs(z_score))
                
                cohens_d = (baseline_acc - current_acc) / np.sqrt((baseline_se**2 + current_se**2) / 2)
                
                if p_value < 0.05 and cohens_d > 0.5:
                    return {
                        "cliff_length": lengths[i],
                        "accuracy_drop": float(baseline_acc - current_acc),
                        "z_score": float(z_score),
                        "p_value": float(p_value),
                        "cohens_d": float(cohens_d),
                        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium"
                    }
        
        return None
    
    def run(self) -> Dict[str, any]:
        """Run context degradation benchmark."""
        print(f"Context lengths: {self.context_lengths}, Samples/length: {self.samples_per_length}")
        print(f"Testing answer positions: {self.answer_positions}")
        
        nq_dataset, tokenized_chunks = self._load_datasets()
        self.tokenized_chunks = tokenized_chunks
        
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=False,
            temperature=1.0
        )
        
        results_by_length_and_position = {}
        
        for position in self.answer_positions:
            print(f"\nTesting position: {position}")
            results_by_length = {}
            accuracies_em = []
            accuracies_f1 = []
            std_errors = []
            
            for ctx_len in self.context_lengths:
                print(f"  {ctx_len} tokens...")
                
                result = self._test_context_length(
                    ctx_len,
                    position,
                    nq_dataset,
                    tokenized_chunks,
                    config
                )
                
                results_by_length[str(ctx_len)] = result
                accuracies_em.append(result['accuracy_em'])
                accuracies_f1.append(result['accuracy_f1'])
                
                se = result['f1_std'] / np.sqrt(result['total']) if result['total'] > 0 else 0.0
                std_errors.append(se)
                
                print(f"    Final: EM={result['accuracy_em']:.3f}, F1={result['accuracy_f1']:.3f}")
            
            results_by_length_and_position[position] = {
                "results_by_length": results_by_length,
                "accuracies_em": accuracies_em,
                "accuracies_f1": accuracies_f1
            }
            
            if len(accuracies_f1) >= 2:
                lengths_array = np.array(self.context_lengths[:len(accuracies_f1)])
                accuracies_array = np.array(accuracies_f1)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    lengths_array,
                    accuracies_array
                )
                
                cliff_point = self._calculate_cliff_point_statistical(
                    self.context_lengths[:len(accuracies_f1)],
                    accuracies_f1,
                    std_errors
                )
                
                results_by_length_and_position[position]["regression"] = {
                    "slope": float(slope),
                    "slope_per_1k_tokens": float(slope * 1000),
                    "intercept": float(intercept),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "std_err": float(std_err),
                    "significant": p_value < 0.05
                }
                
                if cliff_point:
                    results_by_length_and_position[position]["cliff_point"] = cliff_point
        
        position_comparison = {}
        if len(self.answer_positions) > 1:
            for i, pos1 in enumerate(self.answer_positions):
                for pos2 in self.answer_positions[i+1:]:
                    acc1 = results_by_length_and_position[pos1]["accuracies_f1"]
                    acc2 = results_by_length_and_position[pos2]["accuracies_f1"]
                    
                    if len(acc1) == len(acc2):
                        t_stat, p_val = stats.ttest_rel(acc1, acc2)
                        position_comparison[f"{pos1}_vs_{pos2}"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_val),
                            "significant_difference": p_val < 0.05,
                            "mean_diff": float(np.mean(acc1) - np.mean(acc2))
                        }
        
        results = {
            "results_by_position": results_by_length_and_position,
            "position_comparison": position_comparison,
            "context_lengths_tested": self.context_lengths,
            "methodology": {
                "answer_embedding": "natural_wikipedia_passages",
                "context_construction": "token_level_pretokenized",
                "answer_positions_tested": self.answer_positions,
                "cliff_detection": "statistical_significance_with_effect_size",
                "metrics": ["exact_match", "f1_score"]
            }
        }
        
        print("\nPosition comparison:")
        for comparison, stats_data in position_comparison.items():
            print(f"  {comparison}: p={stats_data['p_value']:.4f}, diff={stats_data['mean_diff']:.3f}")
        
        return results