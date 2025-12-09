"""
Context Degradation Benchmark

Measures accuracy degradation as context length increases in quantized models.
"""

import random
import numpy as np
import torch
import gc
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from scipy import stats
import re

from models.model_interface import ModelInterface, GenerationConfig


class ContextDegradationBenchmark:
    """
    Measures accuracy degradation with increasing context length.
    
    Metrics:
        - Accuracy by context length
        - Degradation slope (linear regression)
        - Cliff point (context length with >15% drop)
        - R-squared for linear fit
    """
    
    def __init__(
        self,
        model: ModelInterface,
        context_lengths: List[int] = None,
        samples_per_length: int = 100,
        answer_position_range: Tuple[float, float] = (0.2, 0.8),
        context_tolerance: float = 0.05,
        random_seed: int = 42
    ):
        """
        Initialize context degradation benchmark.
        
        Args:
            model: Model interface instance
            context_lengths: List of context lengths to test (in tokens)
            samples_per_length: Number of samples per length
            answer_position_range: Range for answer position (as fraction of context)
            context_tolerance: Tolerance for context length
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.context_lengths = sorted(context_lengths or [512, 1024, 2048, 4096])
        self.samples_per_length = samples_per_length
        self.answer_position_range = answer_position_range
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
    
    def _build_context_from_tokens(
        self,
        answer_passage: Dict[str, any],
        target_length: int,
        answer_position: float,
        tokenized_chunks: List[Dict[str, any]]
    ) -> Tuple[List[int], int, int]:
        """Build context at exact token length with answer at specified position."""
        context_tokens = []
        
        tokens_before = int(target_length * answer_position)
        tokens_before = max(50, min(tokens_before, target_length - answer_passage['length'] - 50))
        
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
        nq_dataset: any,
        tokenized_chunks: List[Dict[str, any]],
        config: GenerationConfig
    ) -> Dict[str, any]:
        """Test model at specific context length."""
        correct = 0
        total = 0
        attempted = 0
        
        min_length = int(context_length * (1 - self.context_tolerance))
        max_length = int(context_length * (1 + self.context_tolerance))
        
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
            
            answer_position = random.uniform(*self.answer_position_range)
            
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
                    correct += 1
                
                total += 1
                
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
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "correct": correct,
            "total": total,
            "attempted": attempted
        }
    
    def _calculate_cliff_point(
        self,
        lengths: List[int],
        accuracies: List[float],
        threshold: float = 0.15
    ) -> Optional[int]:
        """Identify context length where performance drops significantly."""
        if len(accuracies) < 2:
            return None
        
        baseline = accuracies[0]
        
        for i in range(1, len(accuracies)):
            drop = baseline - accuracies[i]
            if drop >= threshold:
                return lengths[i]
        
        return None
    
    def run(self) -> Dict[str, any]:
        """Run context degradation benchmark."""
        print(f"Context lengths: {self.context_lengths}, Samples/length: {self.samples_per_length}")
        
        nq_dataset, tokenized_chunks = self._load_datasets()
        self.tokenized_chunks = tokenized_chunks
        
        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=False,
            temperature=1.0
        )
        
        results_by_length = {}
        accuracies = []
        
        for ctx_len in self.context_lengths:
            print(f"Testing {ctx_len} tokens...")
            
            result = self._test_context_length(
                ctx_len, 
                nq_dataset, 
                tokenized_chunks, 
                config
            )
            
            accuracy = result['accuracy']
            results_by_length[str(ctx_len)] = result
            accuracies.append(accuracy)
            
            print(f"Accuracy: {accuracy:.3f} ({result['correct']}/{result['total']})")
        
        if len(accuracies) < 2:
            return {
                "results_by_length": results_by_length,
                "error": "insufficient_data"
            }
        
        lengths_array = np.array(self.context_lengths[:len(accuracies)])
        accuracies_array = np.array(accuracies)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            lengths_array, 
            accuracies_array
        )
        
        cliff_point = self._calculate_cliff_point(
            self.context_lengths[:len(accuracies)],
            accuracies,
            threshold=0.15
        )
        
        results = {
            "results_by_length": results_by_length,
            "degradation_slope": float(slope),
            "degradation_slope_per_1k_tokens": float(slope * 1000),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "cliff_point": cliff_point,
            "interpretation": (
                "significant_degradation" if p_value < 0.05 and slope < 0
                else "no_significant_degradation"
            ),
            "context_lengths_tested": self.context_lengths[:len(accuracies)],
            "accuracies": [float(a) for a in accuracies],
            "methodology": {
                "answer_embedding": "natural_wikipedia_passages",
                "context_construction": "token_level_pretokenized",
                "answer_position": "randomized_20_to_80_percent",
                "context_tolerance": f"±{self.context_tolerance*100:.1f}%"
            }
        }
        
        print(f"Slope: {slope * 1000:.4f}/1k tokens, R²: {r_value**2:.4f}, p: {p_value:.4f}")
        if cliff_point:
            print(f"Cliff point: {cliff_point} tokens")
        
        return results