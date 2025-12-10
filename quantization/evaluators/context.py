"""
Context Evaluation Suite - HIGHLY OPTIMIZED

Unified context analysis for RAG models:
1. Context Degradation: Accuracy decline with increasing context length
2. Position Effects: Performance by answer location (start/middle/end)

Key optimizations:
- Pre-tokenized Wikipedia chunks cached once
- Single dataset loading for all lengths and positions
- Efficient context assembly from pre-tokenized pieces
- Shared passage indexing
- Memory-efficient batch processing
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

from models.model_interface import ModelInterface, GenerationConfig


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


class ContextCache:
    """Cached pre-tokenized chunks for efficient context assembly."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.nq_dataset = None
        self.tokenized_chunks = None
        self.answer_index = None
    
    def load(self, max_docs: int = 400):
        """Load and pre-tokenize Wikipedia chunks once."""
        if self.nq_dataset is not None:
            return
        
        print("Loading datasets for context evaluation...")
        self.nq_dataset = load_dataset("nq_open", split="validation")
        print(f"  {len(self.nq_dataset)} questions")
        
        print("  Pre-tokenizing Wikipedia chunks...")
        try:
            wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            self.tokenized_chunks = []
            
            for i, sample in enumerate(wiki):
                if i >= max_docs:
                    break
                
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                
                sentences = re.split(r'[.!?]+\s+', text)
                for sent in sentences:
                    if len(sent.strip()) < 20:
                        continue
                    
                    try:
                        tokens = self.tokenizer.encode(sent.strip() + ". ", add_special_tokens=False)
                        if 10 <= len(tokens) <= 80:
                            self.tokenized_chunks.append({
                                'tokens': tokens,
                                'text': sent.strip() + ". ",
                                'length': len(tokens),
                                'text_lower': sent.strip().lower()
                            })
                    except Exception:
                        continue
                    
                    if len(self.tokenized_chunks) >= 2500:
                        break
                
                if len(self.tokenized_chunks) >= 2500:
                    break
                
                if (i + 1) % 100 == 0:
                    print(f"    {i + 1} docs, {len(self.tokenized_chunks)} chunks...")
        
        except Exception:
            wiki = load_dataset("wikipedia", "20220301.simple", split=f"train[:{max_docs}]")
            self.tokenized_chunks = []
            
            for sample in wiki:
                text = sample.get('text', '')
                if len(text) < 200:
                    continue
                
                sentences = re.split(r'[.!?]+\s+', text)
                for sent in sentences:
                    if len(sent.strip()) < 20:
                        continue
                    
                    try:
                        tokens = self.tokenizer.encode(sent.strip() + ". ", add_special_tokens=False)
                        if 10 <= len(tokens) <= 80:
                            self.tokenized_chunks.append({
                                'tokens': tokens,
                                'text': sent.strip() + ". ",
                                'length': len(tokens),
                                'text_lower': sent.strip().lower()
                            })
                    except Exception:
                        continue
        
        print(f"    {len(self.tokenized_chunks)} chunks pre-tokenized")
        
        print("  Building answer index...")
        self.answer_index = {}
        for idx, chunk in enumerate(self.tokenized_chunks):
            words = chunk['text_lower'].split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                if phrase not in self.answer_index:
                    self.answer_index[phrase] = []
                self.answer_index[phrase].append(idx)
        
        print(f"    {len(self.answer_index)} phrases indexed\n")


class ContextEvaluator:
    """
    Unified context evaluator for length and position effects.
    
    Evaluates:
    - Context Degradation: How does accuracy change with context length?
    - Position Effects: Does answer location matter (start/middle/end)?
    - Statistical Analysis: Regression slopes, cliff points, effect sizes
    """
    
    @staticmethod
    def create_fast(model: ModelInterface):
        """Fast: 25 samples/length, 3 lengths, middle only"""
        return ContextEvaluator(
            model,
            context_lengths=[512, 2048, 4096],
            samples_per_length=25,
            answer_positions=["middle"]
        )
    
    @staticmethod
    def create_standard(model: ModelInterface):
        """Standard: 40 samples/length, 3 lengths, middle only"""
        return ContextEvaluator(
            model,
            context_lengths=[512, 2048, 4096],
            samples_per_length=40,
            answer_positions=["middle"]
        )
    
    @staticmethod
    def create_full(model: ModelInterface):
        """Full: 60 samples/length, 4 lengths, all positions"""
        return ContextEvaluator(
            model,
            context_lengths=[512, 1024, 2048, 4096],
            samples_per_length=60,
            answer_positions=["start", "middle", "end"]
        )
    
    def __init__(
        self,
        model: ModelInterface,
        context_lengths: List[int] = None,
        samples_per_length: int = 40,
        answer_positions: List[str] = None,
        tolerance: float = 0.05
    ):
        self.model = model
        self.cache = ContextCache(model.tokenizer)
        self.context_lengths = sorted(context_lengths or [512, 2048, 4096])
        self.samples_per_length = samples_per_length
        self.answer_positions = answer_positions or ["middle"]
        self.tolerance = tolerance
    
    def _find_answer_chunk(self, answer: str) -> Optional[Dict]:
        """Find chunk containing answer."""
        answer_lower = answer.lower()
        words = answer_lower.split()
        
        if len(words) >= 3:
            phrase = ' '.join(words[:3])
            if phrase in self.cache.answer_index:
                candidates = self.cache.answer_index[phrase]
                if candidates:
                    idx = random.choice(candidates[:10])
                    chunk = self.cache.tokenized_chunks[idx]
                    if len(chunk['tokens']) >= 20:
                        return chunk
        
        for chunk in random.sample(
            self.cache.tokenized_chunks,
            min(100, len(self.cache.tokenized_chunks))
        ):
            if answer_lower in chunk['text_lower'] and len(chunk['tokens']) >= 20:
                return chunk
        
        return None
    
    def _get_position_range(self, position: str) -> Tuple[float, float]:
        """Get fraction range for answer position."""
        if position == "start":
            return (0.0, 0.15)
        elif position == "middle":
            return (0.4, 0.6)
        elif position == "end":
            return (0.85, 1.0)
        return (0.2, 0.8)
    
    def _build_context(
        self,
        answer_chunk: Dict,
        target_length: int,
        position: str
    ) -> Tuple[List[int], int, int]:
        """Build context at exact token length with answer at position."""
        pos_min, pos_max = self._get_position_range(position)
        answer_pos_frac = random.uniform(pos_min, pos_max)
        
        tokens_before = int(target_length * answer_pos_frac)
        tokens_before = max(10, min(tokens_before, target_length - answer_chunk['length'] - 10))
        
        context_tokens = []
        filler_pool = [c for c in self.cache.tokenized_chunks if c != answer_chunk]
        
        current_length = 0
        while current_length < tokens_before:
            chunk = random.choice(filler_pool)
            if current_length + chunk['length'] <= tokens_before:
                context_tokens.extend(chunk['tokens'])
                current_length += chunk['length']
            else:
                remaining = tokens_before - current_length
                if remaining > 5:
                    context_tokens.extend(chunk['tokens'][:remaining])
                break
        
        answer_start = len(context_tokens)
        context_tokens.extend(answer_chunk['tokens'])
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
    
    def _evaluate_length_position(
        self,
        context_length: int,
        position: str,
        config: GenerationConfig
    ) -> Dict[str, any]:
        """Evaluate at specific length and position."""
        correct_em = 0
        f1_scores = []
        total = 0
        attempted = 0
        
        min_len = int(context_length * (1 - self.tolerance))
        max_len = int(context_length * (1 + self.tolerance))
        
        while total < self.samples_per_length and attempted < len(self.cache.nq_dataset):
            sample = self.cache.nq_dataset[attempted]
            attempted += 1
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not question or not answers:
                continue
            
            answer_chunk = self._find_answer_chunk(answers[0])
            if not answer_chunk:
                continue
            
            try:
                context_tokens, ans_start, ans_end = self._build_context(
                    answer_chunk, context_length, position
                )
                
                actual_len = len(context_tokens)
                if not (min_len <= actual_len <= max_len):
                    continue
                
                context_text = self.model.tokenizer.decode(context_tokens, skip_special_tokens=True)
                prompt = f"{context_text}\n\nQuestion: {question}\n\nAnswer:"
                
                output = self.model.generate(prompt, config)
                
                if answers[0].lower() in output.generated_text.lower():
                    correct_em += 1
                
                f1 = max([compute_f1(output.generated_text, ans) for ans in answers])
                f1_scores.append(f1)
                
                total += 1
                
                if total % 5 == 0:
                    current_em = correct_em / total
                    current_f1 = np.mean(f1_scores)
                    print(f"    [{total}/{self.samples_per_length}] "
                          f"EM={current_em:.3f} F1={current_f1:.3f} "
                          f"(attempted {attempted})")
                
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
    
    def _calculate_cliff_point(
        self,
        lengths: List[int],
        accuracies: List[float],
        std_errors: List[float]
    ) -> Optional[Dict[str, any]]:
        """Identify statistically significant performance drop."""
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
        """Run unified context evaluation."""
        print(f"Context Evaluation: {len(self.context_lengths)} lengths x {len(self.answer_positions)} positions\n")
        
        self.cache.load()
        
        config = GenerationConfig(max_new_tokens=15, do_sample=False)
        
        results_by_position = {}
        
        for position in self.answer_positions:
            print(f"Testing position: {position}")
            
            results_by_length = {}
            accuracies_em = []
            accuracies_f1 = []
            std_errors = []
            
            for ctx_len in self.context_lengths:
                print(f"  Length: {ctx_len} tokens")
                
                result = self._evaluate_length_position(ctx_len, position, config)
                
                results_by_length[str(ctx_len)] = result
                accuracies_em.append(result['accuracy_em'])
                accuracies_f1.append(result['accuracy_f1'])
                
                se = result['f1_std'] / np.sqrt(result['total']) if result['total'] > 0 else 0.0
                std_errors.append(se)
                
                print(f"    Final: EM={result['accuracy_em']:.3f}, F1={result['accuracy_f1']:.3f}\n")
            
            results_by_position[position] = {
                "results_by_length": results_by_length,
                "accuracies_em": accuracies_em,
                "accuracies_f1": accuracies_f1
            }
            
            if len(accuracies_f1) >= 2:
                lengths_array = np.array(self.context_lengths[:len(accuracies_f1)])
                accuracies_array = np.array(accuracies_f1)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    lengths_array, accuracies_array
                )
                
                cliff_point = self._calculate_cliff_point(
                    self.context_lengths[:len(accuracies_f1)],
                    accuracies_f1,
                    std_errors
                )
                
                results_by_position[position]["regression"] = {
                    "slope": float(slope),
                    "slope_per_1k_tokens": float(slope * 1000),
                    "intercept": float(intercept),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "std_err": float(std_err),
                    "significant": p_value < 0.05
                }
                
                if cliff_point:
                    results_by_position[position]["cliff_point"] = cliff_point
        
        position_comparison = {}
        if len(self.answer_positions) > 1:
            for i, pos1 in enumerate(self.answer_positions):
                for pos2 in self.answer_positions[i+1:]:
                    acc1 = results_by_position[pos1]["accuracies_f1"]
                    acc2 = results_by_position[pos2]["accuracies_f1"]
                    
                    if len(acc1) == len(acc2):
                        t_stat, p_val = stats.ttest_rel(acc1, acc2)
                        position_comparison[f"{pos1}_vs_{pos2}"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_val),
                            "significant": p_val < 0.05,
                            "mean_diff": float(np.mean(acc1) - np.mean(acc2))
                        }
        
        results = {
            "degradation": {
                "by_position": results_by_position,
                "position_comparison": position_comparison
            },
            "metadata": {
                "context_lengths": self.context_lengths,
                "samples_per_length": self.samples_per_length,
                "answer_positions": self.answer_positions,
                "methodology": {
                    "answer_embedding": "natural_wikipedia_passages",
                    "context_construction": "token_level_pretokenized",
                    "cliff_detection": "statistical_significance_with_effect_size",
                    "metrics": ["exact_match", "f1_score"]
                }
            }
        }
        
        print("\nContext Evaluation Summary:")
        for position in self.answer_positions:
            print(f"\n  Position: {position}")
            if "regression" in results_by_position[position]:
                reg = results_by_position[position]["regression"]
                print(f"    Slope: {reg['slope_per_1k_tokens']:.6f} per 1K tokens")
                print(f"    R²: {reg['r_squared']:.3f}")
            if "cliff_point" in results_by_position[position]:
                cliff = results_by_position[position]["cliff_point"]
                print(f"    Cliff at: {cliff['cliff_length']} tokens (p={cliff['p_value']:.4f})")
        
        if position_comparison:
            print("\n  Position Comparisons:")
            for comp, data in position_comparison.items():
                print(f"    {comp}: diff={data['mean_diff']:.3f}, p={data['p_value']:.4f}")
        
        print()
        return results