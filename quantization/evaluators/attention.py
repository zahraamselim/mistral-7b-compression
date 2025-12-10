"""
Simplified Attention Evaluation for RAG Quantization Research
Resource-efficient implementation focusing on core metrics
"""

import random
import numpy as np
import torch
import gc
from typing import Dict, List, Optional
from datasets import load_dataset
from collections import defaultdict

from models.model_interface import ModelInterface, GenerationConfig


def convert_to_serializable(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
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
    """Simple F1 score between prediction and ground truth."""
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


class AttentionEvaluator:
    """
    Simplified attention evaluator with minimal memory footprint.
    Measures:
    1. Attention Preservation: Does model focus on relevant document?
    2. Attention Drift: Does attention stay stable during generation?
    """
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 50,
        num_documents: int = 3,
        max_length: int = 1024
    ):
        self.model = model
        self.num_samples = num_samples
        self.num_documents = num_documents
        self.max_length = max_length
        
    def _load_data(self):
        """Load minimal dataset."""
        print("Loading Natural Questions dataset...")
        self.nq = load_dataset("nq_open", split="validation[:500]")
        
        print("Loading Wikipedia passages...")
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train")
        wiki = wiki.filter(lambda x: len(x["text"]) > 300)
        
        # Build simple passage index
        self.passages = []
        self.answer_map = defaultdict(list)
        
        for i, item in enumerate(wiki):
            if i >= 200:
                break
            text = item['text'].strip()
            if text:
                self.passages.append(text[:400])
                words = text.lower().split()[:50]
                for j in range(len(words) - 2):
                    phrase = ' '.join(words[j:j+3])
                    self.answer_map[phrase].append(i)
        
        print(f"Loaded {len(self.nq)} questions, {len(self.passages)} passages")
    
    def _find_relevant_passage(self, answer: str) -> Optional[str]:
        """Find passage containing the answer."""
        answer_lower = answer.lower()
        
        # Try phrase matching first
        words = answer_lower.split()
        if len(words) >= 3:
            phrase = ' '.join(words[:3])
            if phrase in self.answer_map:
                indices = self.answer_map[phrase]
                if indices:
                    return self.passages[random.choice(indices[:5])]
        
        # Fallback to substring search
        for passage in random.sample(self.passages, min(50, len(self.passages))):
            if answer_lower in passage.lower():
                return passage
        
        return None
    
    def _select_distractors(self, relevant: str, n: int) -> List[str]:
        """Select distractor passages."""
        distractors = []
        for passage in random.sample(self.passages, min(len(self.passages), n * 3)):
            if passage != relevant and len(distractors) < n:
                distractors.append(passage)
        return distractors[:n]
    
    def _extract_attention(self, attentions, input_ids) -> Optional[np.ndarray]:
        """
        Extract document-level attention from model outputs.
        Uses last layer, averages across heads and tokens.
        """
        if not attentions or len(attentions) == 0:
            return None
        
        try:
            # Get last layer attention
            last_layer = attentions[-1]
            
            # Average across heads: [batch, heads, seq, seq] -> [seq, seq]
            avg_attn = last_layer.mean(dim=1).squeeze()
            
            if avg_attn.dim() == 1:
                # Single sequence
                token_attn = avg_attn.cpu().numpy()
            else:
                # Take attention from last generated token to input
                token_attn = avg_attn[-1, :].cpu().numpy()
            
            # Approximate document boundaries by dividing sequence equally
            seq_len = len(token_attn)
            doc_len = seq_len // self.num_documents
            
            doc_attn = np.zeros(self.num_documents)
            for i in range(self.num_documents):
                start = i * doc_len
                end = (i + 1) * doc_len if i < self.num_documents - 1 else seq_len
                doc_attn[i] = token_attn[start:end].sum()
            
            # Normalize
            total = doc_attn.sum()
            if total > 0:
                doc_attn = doc_attn / total
            
            return doc_attn
        
        except Exception as e:
            print(f"Attention extraction error: {type(e).__name__}")
            return None
    
    def _calculate_metrics(
        self,
        attention: np.ndarray,
        answer_idx: int,
        prev_attention: Optional[np.ndarray]
    ) -> Dict:
        """Calculate preservation and drift metrics."""
        metrics = {}
        
        # Preservation metrics
        top_idx = np.argmax(attention)
        metrics['correct_focus'] = 1.0 if top_idx == answer_idx else 0.0
        
        # Rank of answer document
        sorted_indices = np.argsort(attention)[::-1]
        rank = int(np.where(sorted_indices == answer_idx)[0][0] + 1)
        metrics['answer_rank'] = rank
        
        # Attention on answer document
        metrics['answer_attention'] = float(attention[answer_idx])
        
        # Drift metric (if we have previous attention)
        if prev_attention is not None:
            drift = np.abs(attention - prev_attention).sum()
            metrics['drift'] = float(drift)
            
            # Drift specifically on answer document
            answer_drift = abs(attention[answer_idx] - prev_attention[answer_idx])
            metrics['answer_drift'] = float(answer_drift)
        
        return metrics
    
    def run(self) -> Dict:
        """Run simplified evaluation."""
        print(f"Running attention evaluation with {self.num_samples} samples")
        print(f"Documents per sample: {self.num_documents}")
        print(f"Max length: {self.max_length}\n")
        
        self._load_data()
        
        # Collect metrics
        correct_focus_scores = []
        answer_ranks = []
        answer_attention_scores = []
        drift_scores = []
        answer_drift_scores = []
        f1_scores = []
        
        successful = 0
        attempted = 0
        
        config = GenerationConfig(max_new_tokens=20, do_sample=False)
        
        while successful < self.num_samples and attempted < len(self.nq):
            sample = self.nq[attempted]
            attempted += 1
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not question or not answers:
                continue
            
            # Find relevant passage
            relevant = self._find_relevant_passage(answers[0])
            if not relevant:
                continue
            
            # Get distractors
            distractors = self._select_distractors(relevant, self.num_documents - 1)
            if len(distractors) < self.num_documents - 1:
                continue
            
            # Shuffle documents
            all_docs = [relevant] + distractors
            random.shuffle(all_docs)
            answer_idx = all_docs.index(relevant)
            
            # Create prompt
            doc_text = "\n\n".join([
                f"Document {i+1}: {doc}" 
                for i, doc in enumerate(all_docs)
            ])
            prompt = f"Question: {question}\n\n{doc_text}\n\nAnswer:"
            
            # Truncate if needed
            tokens = self.model.tokenizer.encode(prompt)
            if len(tokens) > self.max_length:
                prompt = self.model.tokenizer.decode(tokens[:self.max_length])
            
            try:
                # Generate with attention tracking
                output = self.model.generate(prompt, config, return_attentions=True)
                
                if not output.attentions:
                    continue
                
                # Extract attention at two points for drift measurement
                inputs = self.model.encode(prompt, max_length=self.max_length)
                
                # Initial attention (after prompt)
                initial_attn = self._extract_attention(
                    output.attentions[:1], 
                    inputs['input_ids']
                )
                
                # Final attention (after generation)
                final_attn = self._extract_attention(
                    output.attentions[-1:],
                    inputs['input_ids']
                )
                
                if initial_attn is None or final_attn is None:
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(final_attn, answer_idx, initial_attn)
                
                # Quality metric
                f1 = max([compute_f1(output.generated_text, ans) for ans in answers])
                
                # Store results
                correct_focus_scores.append(metrics['correct_focus'])
                answer_ranks.append(metrics['answer_rank'])
                answer_attention_scores.append(metrics['answer_attention'])
                if 'drift' in metrics:
                    drift_scores.append(metrics['drift'])
                    answer_drift_scores.append(metrics['answer_drift'])
                f1_scores.append(f1)
                
                successful += 1
                
                if successful % 10 == 0:
                    print(f"Progress: {successful}/{self.num_samples} "
                          f"(tried {attempted}) "
                          f"Precision@1={np.mean(correct_focus_scores):.3f} "
                          f"F1={np.mean(f1_scores):.3f}")
                
                # Memory cleanup
                del output, inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at sample {attempted}, cleaning up...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"Error at sample {attempted}: {type(e).__name__}")
                continue
        
        if successful == 0:
            raise ValueError("No successful evaluations completed")
        
        # Compile results
        results = {
            "preservation": {
                "precision_at_1": np.mean(correct_focus_scores),
                "mean_rank": np.mean(answer_ranks),
                "median_rank": np.median(answer_ranks),
                "mean_attention_on_answer": np.mean(answer_attention_scores)
            },
            "drift": {
                "mean_total_drift": np.mean(drift_scores) if drift_scores else 0.0,
                "mean_answer_drift": np.mean(answer_drift_scores) if answer_drift_scores else 0.0
            },
            "quality": {
                "mean_f1": np.mean(f1_scores)
            },
            "metadata": {
                "successful_samples": successful,
                "attempted_samples": attempted,
                "num_documents": self.num_documents,
                "max_length": self.max_length
            }
        }
        
        # Convert to JSON-serializable types
        results = convert_to_serializable(results)
        
        print(f"\nEvaluation complete:")
        print(f"Samples: {successful}/{attempted}")
        print(f"Precision@1: {results['preservation']['precision_at_1']:.3f}")
        print(f"Mean Rank: {results['preservation']['mean_rank']:.2f}")
        print(f"Attention Drift: {results['drift']['mean_total_drift']:.4f}")
        print(f"Answer Drift: {results['drift']['mean_answer_drift']:.4f}")
        print(f"F1 Score: {results['quality']['mean_f1']:.3f}\n")
        
        return results


# Usage
if __name__ == "__main__":
    evaluator = SimpleAttentionEvaluator(
        model=model,
        num_samples=50,
        num_documents=3,
        max_length=1024
    )
    
    results = evaluator.run()
    
    # Save results
    import json
    with open("attention_results.json", "w") as f:
        json.dump(results, f, indent=2)