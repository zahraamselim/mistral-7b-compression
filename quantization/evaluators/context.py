"""
Simplified Context Length Evaluation for RAG Quantization Research
Resource-efficient implementation with controlled context assembly
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
    """Simple F1 score."""
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
    Simplified context length evaluator.
    Tests performance degradation as context increases.
    """
    
    def __init__(
        self,
        model: ModelInterface,
        context_lengths: List[int] = None,
        samples_per_length: int = 20
    ):
        self.model = model
        self.context_lengths = context_lengths or [512, 1024, 2048]
        self.samples_per_length = samples_per_length
    
    def _load_data(self):
        """Load datasets."""
        print("Loading SQuAD for questions and answers...")
        self.squad = load_dataset("squad_v2", split="validation[:300]")
        
        print("Loading Wikitext for filler passages...")
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
        self.wiki_passages = [
            item['text'].strip() 
            for item in wiki 
            if len(item['text'].strip()) > 100
        ][:500]
        
        print(f"Loaded {len(self.squad)} questions, {len(self.wiki_passages)} passages")
    
    def _build_context(
        self,
        question: str,
        context: str,
        target_length: int
    ) -> str:
        """Build context padded to target token length."""
        
        # Start with question and answer context
        base_text = f"{context}\n\nQuestion: {question}"
        base_tokens = self.model.tokenizer.encode(base_text)
        
        if len(base_tokens) >= target_length:
            # Truncate if too long
            truncated = self.model.tokenizer.decode(base_tokens[:target_length])
            return truncated
        
        # Add filler passages until we reach target length
        filler_parts = [base_text]
        current_length = len(base_tokens)
        
        available_passages = random.sample(
            self.wiki_passages, 
            min(50, len(self.wiki_passages))
        )
        
        for passage in available_passages:
            if current_length >= target_length:
                break
            
            # Limit passage length
            passage_tokens = self.model.tokenizer.encode(passage)[:200]
            
            if current_length + len(passage_tokens) <= target_length:
                filler_parts.append(passage)
                current_length += len(passage_tokens)
        
        combined = "\n\n".join(filler_parts) + "\n\nAnswer:"
        
        # Final truncation to exact length
        final_tokens = self.model.tokenizer.encode(combined)[:target_length]
        return self.model.tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    def _evaluate_at_length(
        self,
        context_length: int,
        config: GenerationConfig
    ) -> Dict:
        """Evaluate at specific context length."""
        print(f"\nEvaluating at {context_length} tokens...")
        
        correct = 0
        f1_scores = []
        total = 0
        
        for sample in self.squad:
            if total >= self.samples_per_length:
                break
            
            question = sample.get('question', '')
            context = sample.get('context', '')
            answers = sample.get('answers', {}).get('text', [])
            
            if not question or not answers or not context:
                continue
            
            try:
                # Build context at target length
                full_context = self._build_context(question, context, context_length)
                
                # Verify length is close to target
                actual_length = len(self.model.tokenizer.encode(full_context))
                if actual_length < context_length * 0.9:
                    continue
                
                # Generate answer
                output = self.model.generate(full_context, config)
                
                # Check correctness
                response = output.generated_text.lower()
                answer_lower = answers[0].lower()
                
                if answer_lower in response:
                    correct += 1
                
                f1 = max([compute_f1(output.generated_text, ans) for ans in answers])
                f1_scores.append(f1)
                
                total += 1
                
                if total % 5 == 0:
                    print(f"  Progress: {total}/{self.samples_per_length} "
                          f"Accuracy={correct/total:.3f} "
                          f"F1={np.mean(f1_scores):.3f}")
                
                # Memory cleanup
                del output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at length {context_length}, stopping this length")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                continue
            except Exception as e:
                print(f"  Error: {type(e).__name__}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        mean_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        print(f"  Final: Accuracy={accuracy:.3f}, F1={mean_f1:.3f}, Samples={total}")
        
        return {
            "accuracy": accuracy,
            "f1": mean_f1,
            "num_samples": total
        }
    
    def run(self) -> Dict:
        """Run context evaluation across all lengths."""
        print(f"Context Evaluation: {len(self.context_lengths)} lengths")
        print(f"Samples per length: {self.samples_per_length}\n")
        
        self._load_data()
        
        config = GenerationConfig(max_new_tokens=15, do_sample=False)
        
        results_by_length = {}
        accuracies = []
        f1_scores = []
        
        for length in self.context_lengths:
            result = self._evaluate_at_length(length, config)
            
            results_by_length[str(length)] = result
            accuracies.append(result['accuracy'])
            f1_scores.append(result['f1'])
        
        # Calculate degradation slope
        slope = 0.0
        r_squared = 0.0
        
        if len(f1_scores) >= 2:
            lengths_array = np.array(self.context_lengths)
            f1_array = np.array(f1_scores)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                lengths_array, f1_array
            )
            r_squared = r_value ** 2
            
            print(f"\nRegression Analysis:")
            print(f"  Slope: {slope:.6f} per token")
            print(f"  Slope per 1K tokens: {slope * 1000:.6f}")
            print(f"  R-squared: {r_squared:.3f}")
        
        results = {
            "by_length": results_by_length,
            "regression": {
                "slope": slope,
                "slope_per_1k_tokens": slope * 1000,
                "r_squared": r_squared
            },
            "metadata": {
                "context_lengths": self.context_lengths,
                "samples_per_length": self.samples_per_length
            }
        }
        
        # Convert to JSON-serializable
        results = convert_to_serializable(results)
        
        print(f"\nContext evaluation complete")
        print(f"Degradation slope: {results['regression']['slope_per_1k_tokens']:.6f} per 1K tokens\n")
        
        return results


# Usage
if __name__ == "__main__":
    evaluator = SimpleContextEvaluator(
        model=model,
        context_lengths=[512, 1024, 2048],
        samples_per_length=20
    )
    
    results = evaluator.run()
    
    # Save results
    import json
    with open("context_results.json", "w") as f:
        json.dump(results, f, indent=2)