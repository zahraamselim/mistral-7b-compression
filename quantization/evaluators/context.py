"""
Context Length Evaluation - Optimized for Kaggle T4 (16GB VRAM)
Measures performance degradation as context length increases
"""

import random
import numpy as np
import torch
import gc
from typing import Dict, List, Tuple
from datasets import load_dataset
from scipy import stats

from models.model_interface import ModelInterface, GenerationConfig


def convert_to_serializable(obj):
    """Convert NumPy types to native Python types."""
    if isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
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


class ContextEvaluator:
    """
    Optimized context length evaluator for Mistral 7B on T4.
    
    Measures:
    - Performance degradation as context increases
    - Answer position sensitivity (needle in haystack)
    - Context utilization efficiency
    
    Strategy:
    - Test multiple context lengths (512, 1024, 2048, 4096)
    - Place answer at different positions (start, middle, end)
    - Use real Q&A data with controlled filler passages
    """
    
    def __init__(
        self,
        model: ModelInterface,
        context_lengths: List[int] = None,
        samples_per_length: int = 25,
        test_positions: List[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: ModelInterface instance
            context_lengths: Context lengths to test
            samples_per_length: Samples per length
            test_positions: Answer positions to test ['start', 'middle', 'end']
        """
        self.model = model
        self.context_lengths = context_lengths or [512, 1024, 2048, 4096]
        self.samples_per_length = samples_per_length
        self.test_positions = test_positions or ['middle']  # Default to middle only
        
        self._cleanup()
    
    def _cleanup(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _load_data(self):
        """Load datasets."""
        print("Loading SQuAD for Q&A...")
        self.squad = load_dataset("squad_v2", split="validation[:500]")
        
        print("Loading WikiText for filler passages...")
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train[:2000]")
        self.wiki_passages = [
            item['text'].strip() 
            for item in wiki 
            if len(item['text'].strip()) > 100
        ][:800]
        
        print(f"Loaded {len(self.squad)} questions, {len(self.wiki_passages)} filler passages")
    
    def _build_context_at_position(
        self,
        answer_context: str,
        question: str,
        target_length: int,
        position: str = 'middle'
    ) -> Tuple[str, int]:
        """
        Build context with answer at specific position.
        
        Args:
            answer_context: Context containing the answer
            question: Question text
            target_length: Target token length
            position: 'start', 'middle', or 'end'
            
        Returns:
            (full_context, answer_position_tokens)
        """
        # Tokenize answer context
        answer_tokens = self.model.tokenizer.encode(answer_context)
        answer_length = len(answer_tokens)
        
        if answer_length >= target_length:
            # Answer context is already long enough
            truncated = self.model.tokenizer.decode(answer_tokens[:target_length])
            return truncated, 0
        
        # Calculate filler needed
        tokens_needed = target_length - answer_length
        
        # Determine position
        if position == 'start':
            before_tokens = 0
            after_tokens = tokens_needed
        elif position == 'end':
            before_tokens = tokens_needed
            after_tokens = 0
        else:  # middle
            before_tokens = tokens_needed // 2
            after_tokens = tokens_needed - before_tokens
        
        # Build filler sections
        before_filler = self._build_filler(before_tokens) if before_tokens > 0 else ""
        after_filler = self._build_filler(after_tokens) if after_tokens > 0 else ""
        
        # Combine
        parts = []
        if before_filler:
            parts.append(before_filler)
        parts.append(answer_context)
        if after_filler:
            parts.append(after_filler)
        
        combined = "\n\n".join(parts) + f"\n\nQuestion: {question}\nAnswer:"
        
        # Verify and truncate to exact length
        final_tokens = self.model.tokenizer.encode(combined)[:target_length]
        final_context = self.model.tokenizer.decode(final_tokens, skip_special_tokens=True)
        
        return final_context, before_tokens
    
    def _build_filler(self, num_tokens: int) -> str:
        """Build filler text of approximately num_tokens length."""
        if num_tokens <= 0:
            return ""
        
        filler_parts = []
        current_tokens = 0
        
        available = random.sample(
            self.wiki_passages,
            min(len(self.wiki_passages), num_tokens // 50 + 10)
        )
        
        for passage in available:
            if current_tokens >= num_tokens:
                break
            
            # Limit each passage
            passage_tokens = self.model.tokenizer.encode(passage)[:150]
            filler_parts.append(self.model.tokenizer.decode(passage_tokens))
            current_tokens += len(passage_tokens)
        
        return "\n\n".join(filler_parts)
    
    def _evaluate_at_length(
        self,
        context_length: int,
        position: str,
        config: GenerationConfig
    ) -> Dict:
        """Evaluate at specific context length and position."""
        print(f"\n  Testing at {context_length} tokens (answer at {position})...")
        
        results = {
            'correct': 0,
            'f1_scores': [],
            'total': 0,
            'oom_count': 0,
            'avg_answer_position': []
        }
        
        for sample in self.squad:
            if results['total'] >= self.samples_per_length:
                break
            
            question = sample.get('question', '')
            context = sample.get('context', '')
            answers = sample.get('answers', {}).get('text', [])
            
            if not question or not answers or not context:
                continue
            
            try:
                # Build context at target length with answer at position
                full_context, answer_pos = self._build_context_at_position(
                    context, question, context_length, position
                )
                
                # Verify length
                actual_length = len(self.model.tokenizer.encode(full_context))
                if actual_length < context_length * 0.85:  # Too short
                    continue
                
                # Generate answer
                output = self.model.generate(full_context, config)
                
                # Evaluate
                response = output.generated_text.lower().strip()
                answer_lower = answers[0].lower()
                
                if answer_lower in response:
                    results['correct'] += 1
                
                f1 = max([compute_f1(output.generated_text, ans) for ans in answers])
                results['f1_scores'].append(f1)
                results['avg_answer_position'].append(answer_pos)
                results['total'] += 1
                
                # Progress
                if results['total'] % 5 == 0:
                    acc = results['correct'] / results['total']
                    avg_f1 = np.mean(results['f1_scores'])
                    print(f"    Progress: {results['total']}/{self.samples_per_length} | "
                          f"Acc: {acc:.3f} | F1: {avg_f1:.3f}")
                
                # Cleanup
                del output
                if results['total'] % 5 == 0:
                    self._cleanup()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results['oom_count'] += 1
                    print(f"    OOM at {context_length} tokens, stopping this configuration")
                    self._cleanup()
                    break
                continue
            except Exception:
                continue
        
        # Calculate metrics
        accuracy = results['correct'] / results['total'] if results['total'] > 0 else 0.0
        mean_f1 = np.mean(results['f1_scores']) if results['f1_scores'] else 0.0
        
        print(f"    Results: Acc={accuracy:.3f}, F1={mean_f1:.3f}, "
              f"Samples={results['total']}, OOM={results['oom_count']}")
        
        return {
            'accuracy': accuracy,
            'f1': mean_f1,
            'num_samples': results['total'],
            'oom_count': results['oom_count'],
            'answer_position': position
        }
    
    def run(self) -> Dict:
        """Run context evaluation."""
        print(f"Context lengths: {self.context_lengths}")
        print(f"Samples per length: {self.samples_per_length}")
        print(f"Answer positions: {self.test_positions}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Initial VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print()
        
        self._load_data()
        
        config = GenerationConfig(max_new_tokens=20, do_sample=False)
        
        # Storage for results
        results_by_length = {}
        results_by_position = {}
        
        all_f1_scores = []
        all_lengths = []
        
        # Test each length and position
        for length in self.context_lengths:
            results_by_length[str(length)] = {}
            
            for position in self.test_positions:
                result = self._evaluate_at_length(length, position, config)
                
                results_by_length[str(length)][position] = result
                
                # Track for regression
                all_lengths.append(length)
                all_f1_scores.append(result['f1'])
                
                # Track by position
                if position not in results_by_position:
                    results_by_position[position] = []
                results_by_position[position].append({
                    'length': length,
                    'f1': result['f1'],
                    'accuracy': result['accuracy']
                })
        
        # Calculate degradation metrics
        slope = 0.0
        r_squared = 0.0
        
        if len(all_f1_scores) >= 2:
            lengths_array = np.array(all_lengths)
            f1_array = np.array(all_f1_scores)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                lengths_array, f1_array
            )
            r_squared = r_value ** 2
            
            print(f"\nDegradation Analysis:")
            print(f"  Slope: {slope:.6f} per token")
            print(f"  Slope per 1K tokens: {slope * 1000:.4f}")
            print(f"  R-squared: {r_squared:.3f}")
        
        # Position analysis
        position_summary = {}
        for position, data in results_by_position.items():
            f1_values = [d['f1'] for d in data]
            position_summary[position] = {
                'mean_f1': float(np.mean(f1_values)),
                'std_f1': float(np.std(f1_values)),
                'min_f1': float(np.min(f1_values)),
                'max_f1': float(np.max(f1_values))
            }
        
        results = {
            "by_length": results_by_length,
            "by_position": position_summary,
            "degradation": {
                "slope_per_token": slope,
                "slope_per_1k_tokens": slope * 1000,
                "r_squared": r_squared,
                "interpretation": self._interpret_slope(slope * 1000)
            },
            "metadata": {
                "context_lengths": self.context_lengths,
                "samples_per_length": self.samples_per_length,
                "positions_tested": self.test_positions
            }
        }
        
        results = convert_to_serializable(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _interpret_slope(self, slope_per_1k: float) -> str:
        """Interpret degradation slope."""
        if abs(slope_per_1k) < 0.001:
            return "negligible"
        elif abs(slope_per_1k) < 0.01:
            return "minimal"
        elif abs(slope_per_1k) < 0.05:
            return "moderate"
        else:
            return "significant"
    
    def _print_summary(self, results: Dict):
        """Print formatted results."""
        print("\nPERFORMANCE BY LENGTH:")
        for length, data in results['by_length'].items():
            print(f"  {length} tokens:")
            for position, metrics in data.items():
                print(f"    {position}: F1={metrics['f1']:.3f}, "
                      f"Acc={metrics['accuracy']:.3f}, "
                      f"N={metrics['num_samples']}")
        
        if results['by_position']:
            print("\nPERFORMANCE BY POSITION:")
            for position, stats in results['by_position'].items():
                print(f"  {position}: mean={stats['mean_f1']:.3f}, "
                      f"std={stats['std_f1']:.3f}")
        
        print("\nDEGRADATION ANALYSIS:")
        deg = results['degradation']
        print(f"  Slope: {deg['slope_per_1k_tokens']:.4f} per 1K tokens")
        print(f"  R²: {deg['r_squared']:.3f}")
        print(f"  Interpretation: {deg['interpretation']}")