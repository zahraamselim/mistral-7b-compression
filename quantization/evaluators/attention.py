"""
Attention Evaluation - Optimized for Kaggle T4 (16GB VRAM)
Measures attention preservation and drift during RAG generation
"""

import random
import numpy as np
import torch
import gc
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from collections import defaultdict

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
    Optimized attention evaluator for Mistral 7B on T4 GPU.
    
    Measures:
    - Attention preservation: Does model focus on correct document?
    - Attention drift: How much does focus shift during generation?
    - Quality: Accuracy of generated answers
    
    Memory-efficient strategy:
    - Context: 512 tokens (safe for T4, fits ~5 documents)
    - Generation: 20 tokens (enough for answer + drift measurement)
    - Drift tracking: start, middle, end of generation
    - Batch size: 1 with immediate cleanup
    """
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 50,
        num_documents: int = 5,
        max_context_length: int = 512,
        generation_length: int = 20
    ):
        """
        Initialize evaluator.
        
        Args:
            model: ModelInterface instance
            num_samples: Number of test samples
            num_documents: Documents per sample (1 relevant + N-1 distractors)
            max_context_length: Max tokens for context
            generation_length: Tokens to generate for answer
        """
        self.model = model
        self.num_samples = num_samples
        self.num_documents = num_documents
        self.max_context_length = max_context_length
        self.generation_length = generation_length
        
        # Force cleanup
        self._cleanup()
        
    def _cleanup(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _load_data(self):
        """Load datasets efficiently."""
        print("Loading datasets...")
        
        # Natural Questions for Q&A pairs
        self.nq = load_dataset("nq_open", split="validation[:500]")
        
        # Wikipedia for passages
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train[:2000]")
        wiki = wiki.filter(lambda x: len(x["text"]) > 200)
        
        # Build passage index
        self.passages = []
        self.answer_map = defaultdict(list)
        
        for i, item in enumerate(wiki):
            text = item['text'].strip()
            if text and len(text) > 200:
                # Use 250 char passages
                passage = text[:250].strip()
                self.passages.append(passage)
                
                # Index by 3-gram phrases for efficient lookup
                words = text.lower().split()[:50]
                for j in range(min(len(words) - 2, 30)):
                    phrase = ' '.join(words[j:j+3])
                    self.answer_map[phrase].append(len(self.passages) - 1)
            
            if len(self.passages) >= 200:
                break
        
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
        candidates = random.sample(self.passages, min(50, len(self.passages)))
        for passage in candidates:
            if answer_lower in passage.lower():
                return passage
        
        return None
    
    def _select_distractors(self, relevant: str, n: int) -> List[str]:
        """Select distractor passages."""
        distractors = []
        candidates = [p for p in self.passages if p != relevant]
        selected = random.sample(candidates, min(len(candidates), n * 3))
        
        for passage in selected:
            if len(distractors) >= n:
                break
            distractors.append(passage)
        
        return distractors[:n]
    
    def _extract_attention_snapshot(
        self,
        attentions: List,
        step_idx: int,
        input_length: int
    ) -> Optional[np.ndarray]:
        """
        Extract attention distribution at a specific generation step.
        
        Args:
            attentions: Model attention outputs
            step_idx: Generation step index
            input_length: Length of input context
            
        Returns:
            Document-level attention distribution or None
        """
        if not attentions or step_idx >= len(attentions):
            return None
        
        try:
            step_attn = attentions[step_idx]
            
            # Get last layer
            if isinstance(step_attn, (list, tuple)):
                last_layer = step_attn[-1]
            else:
                last_layer = step_attn
            
            if not isinstance(last_layer, torch.Tensor):
                return None
            
            # Move to CPU immediately
            last_layer = last_layer.detach().cpu()
            
            # Average across batch and heads: [batch, heads, seq, seq] -> [seq, seq]
            while last_layer.dim() > 2:
                last_layer = last_layer.mean(dim=0)
            
            if last_layer.dim() != 2:
                return None
            
            # Get attention from last generated token to input tokens
            token_attn = last_layer[-1, :].numpy()
            
            # Only use input context (exclude generated tokens)
            token_attn = token_attn[:input_length]
            
            # Aggregate by document
            doc_len = input_length // self.num_documents
            if doc_len < 5:  # Need reasonable tokens per doc
                return None
            
            doc_attn = np.zeros(self.num_documents)
            for i in range(self.num_documents):
                start = i * doc_len
                end = (i + 1) * doc_len if i < self.num_documents - 1 else input_length
                if end > start:
                    doc_attn[i] = token_attn[start:end].sum()
            
            # Normalize
            total = doc_attn.sum()
            if total > 1e-8:
                doc_attn = doc_attn / total
            else:
                return None
            
            return doc_attn
            
        except Exception:
            return None
    
    def _calculate_metrics(
        self,
        attention_snapshots: Dict[str, np.ndarray],
        answer_idx: int
    ) -> Dict:
        """
        Calculate preservation and drift metrics.
        
        Args:
            attention_snapshots: Dict with 'start', 'mid', 'end' attention
            answer_idx: Index of correct document
            
        Returns:
            Metrics dict
        """
        metrics = {}
        
        # Use final attention for preservation
        final_attn = attention_snapshots.get('end')
        if final_attn is None:
            return {}
        
        # Preservation: correct document focus
        top_idx = np.argmax(final_attn)
        metrics['correct_focus'] = 1.0 if top_idx == answer_idx else 0.0
        
        # Answer document rank
        sorted_indices = np.argsort(final_attn)[::-1]
        rank = int(np.where(sorted_indices == answer_idx)[0][0] + 1)
        metrics['answer_rank'] = rank
        
        # Answer attention weight
        metrics['answer_attention'] = float(final_attn[answer_idx])
        
        # Attention concentration (Gini coefficient)
        sorted_attn = np.sort(final_attn)
        n = len(sorted_attn)
        cumsum = np.cumsum(sorted_attn)
        if cumsum[-1] > 0:
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        else:
            gini = 0.0
        metrics['gini'] = float(gini)
        
        # Drift metrics
        start_attn = attention_snapshots.get('start')
        if start_attn is not None:
            # Total drift (L1 distance)
            total_drift = np.abs(final_attn - start_attn).sum()
            metrics['total_drift'] = float(total_drift)
            
            # Answer document drift
            answer_drift = abs(final_attn[answer_idx] - start_attn[answer_idx])
            metrics['answer_drift'] = float(answer_drift)
            
            # Max single-document drift
            max_drift = np.abs(final_attn - start_attn).max()
            metrics['max_drift'] = float(max_drift)
            
            # Drift direction: did attention to answer increase or decrease?
            drift_direction = final_attn[answer_idx] - start_attn[answer_idx]
            metrics['answer_drift_direction'] = float(drift_direction)
        
        return metrics
    
    def run(self) -> Dict:
        """Run attention evaluation."""
        print(f"Target samples: {self.num_samples}")
        print(f"Documents per sample: {self.num_documents}")
        print(f"Max context: {self.max_context_length} tokens")
        print(f"Generation: {self.generation_length} tokens")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Initial VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print()
        
        self._load_data()
        
        # Storage for results
        metrics_list = []
        f1_scores = []
        exact_matches = []
        
        successful = 0
        attempted = 0
        skipped = 0
        oom_count = 0
        
        config = GenerationConfig(
            max_new_tokens=self.generation_length,
            do_sample=False
        )
        
        # Try up to 3x target to account for failures
        max_attempts = min(len(self.nq), self.num_samples * 3)
        
        while successful < self.num_samples and attempted < max_attempts:
            sample = self.nq[attempted]
            attempted += 1
            
            # Periodic cleanup
            if attempted % 10 == 0:
                self._cleanup()
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not question or not answers:
                skipped += 1
                continue
            
            # Find relevant passage
            relevant = self._find_relevant_passage(answers[0])
            if not relevant:
                skipped += 1
                continue
            
            # Get distractors
            distractors = self._select_distractors(relevant, self.num_documents - 1)
            if len(distractors) < self.num_documents - 1:
                skipped += 1
                continue
            
            # Shuffle documents
            all_docs = [relevant] + distractors
            random.shuffle(all_docs)
            answer_idx = all_docs.index(relevant)
            
            # Create prompt
            doc_texts = [f"Document {i+1}: {doc}" for i, doc in enumerate(all_docs)]
            context = "\n\n".join(doc_texts)
            prompt = f"Question: {question}\n\n{context}\n\nAnswer:"
            
            # Truncate to max length
            tokens = self.model.tokenizer.encode(prompt, add_special_tokens=True)
            if len(tokens) > self.max_context_length:
                tokens = tokens[:self.max_context_length]
                prompt = self.model.tokenizer.decode(tokens, skip_special_tokens=True)
            
            input_length = len(tokens)
            
            try:
                # Generate with attention tracking
                output = self.model.generate(prompt, config, return_attentions=True)
                
                if not output.attentions or len(output.attentions) < 3:
                    skipped += 1
                    continue
                
                # Extract attention snapshots
                num_steps = len(output.attentions)
                snapshots = {}
                
                # Start
                start_attn = self._extract_attention_snapshot(
                    output.attentions, 0, input_length
                )
                if start_attn is not None:
                    snapshots['start'] = start_attn
                
                # Middle
                mid_idx = num_steps // 2
                mid_attn = self._extract_attention_snapshot(
                    output.attentions, mid_idx, input_length
                )
                if mid_attn is not None:
                    snapshots['mid'] = mid_attn
                
                # End
                end_attn = self._extract_attention_snapshot(
                    output.attentions, -1, input_length
                )
                if end_attn is not None:
                    snapshots['end'] = end_attn
                
                if 'end' not in snapshots:
                    skipped += 1
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(snapshots, answer_idx)
                if not metrics:
                    skipped += 1
                    continue
                
                # Quality metrics
                pred_text = output.generated_text.strip().lower()
                f1 = max([compute_f1(pred_text, ans.lower()) for ans in answers])
                exact_match = 1.0 if any(ans.lower() in pred_text for ans in answers) else 0.0
                
                metrics_list.append(metrics)
                f1_scores.append(f1)
                exact_matches.append(exact_match)
                
                successful += 1
                
                # Progress update
                if successful % 10 == 0:
                    avg_precision = np.mean([m['correct_focus'] for m in metrics_list])
                    avg_f1 = np.mean(f1_scores)
                    print(f"{successful}/{self.num_samples} | "
                          f"Tried: {attempted} | "
                          f"P@1: {avg_precision:.3f} | "
                          f"F1: {avg_f1:.3f}")
                
                # Cleanup
                del output, snapshots, metrics
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    self._cleanup()
                skipped += 1
                continue
            except Exception:
                skipped += 1
                continue
        
        # Final cleanup
        self._cleanup()
        
        if successful == 0:
            raise ValueError(
                f"Failed to complete any evaluations.\n"
                f"Attempted: {attempted}, Skipped: {skipped}, OOM: {oom_count}\n"
                f"Try reducing max_context_length or num_documents"
            )
        
        # Aggregate results
        results = {
            "preservation": {
                "precision_at_1": float(np.mean([m['correct_focus'] for m in metrics_list])),
                "mean_rank": float(np.mean([m['answer_rank'] for m in metrics_list])),
                "median_rank": float(np.median([m['answer_rank'] for m in metrics_list])),
                "mean_attention_on_answer": float(np.mean([m['answer_attention'] for m in metrics_list])),
                "attention_concentration": float(np.mean([m['gini'] for m in metrics_list]))
            },
            "drift": {
                "mean_total_drift": float(np.mean([m['total_drift'] for m in metrics_list if 'total_drift' in m])) if any('total_drift' in m for m in metrics_list) else 0.0,
                "mean_answer_drift": float(np.mean([m['answer_drift'] for m in metrics_list if 'answer_drift' in m])) if any('answer_drift' in m for m in metrics_list) else 0.0,
                "mean_max_drift": float(np.mean([m['max_drift'] for m in metrics_list if 'max_drift' in m])) if any('max_drift' in m for m in metrics_list) else 0.0,
                "mean_answer_drift_direction": float(np.mean([m['answer_drift_direction'] for m in metrics_list if 'answer_drift_direction' in m])) if any('answer_drift_direction' in m for m in metrics_list) else 0.0
            },
            "quality": {
                "exact_match": float(np.mean(exact_matches)),
                "f1_mean": float(np.mean(f1_scores))
            },
            "metadata": {
                "successful_samples": successful,
                "attempted_samples": attempted,
                "skipped_samples": skipped,
                "oom_errors": oom_count,
                "num_documents": self.num_documents,
                "max_context_length": self.max_context_length,
                "generation_length": self.generation_length
            }
        }
        
        results = convert_to_serializable(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print formatted results."""
        
        meta = results['metadata']
        print(f"Successful: {meta['successful_samples']}/{meta['attempted_samples']}")
        print(f"Skipped: {meta['skipped_samples']} | OOM: {meta['oom_errors']}")
        
        print(f"\nPRESERVATION METRICS:")
        pres = results['preservation']
        print(f"  Precision@1: {pres['precision_at_1']:.3f}")
        print(f"  Mean Rank: {pres['mean_rank']:.2f}")
        print(f"  Median Rank: {pres['median_rank']:.1f}")
        print(f"  Attention on Answer: {pres['mean_attention_on_answer']:.3f}")
        print(f"  Concentration (Gini): {pres['attention_concentration']:.3f}")
        
        print(f"\nDRIFT METRICS:")
        drift = results['drift']
        print(f"  Total Drift: {drift['mean_total_drift']:.4f}")
        print(f"  Answer Drift: {drift['mean_answer_drift']:.4f}")
        print(f"  Max Drift: {drift['mean_max_drift']:.4f}")
        print(f"  Answer Direction: {drift['mean_answer_drift_direction']:+.4f}")
        
        print(f"\nQUALITY METRICS:")
        qual = results['quality']
        print(f"  Exact Match: {qual['exact_match']:.3f}")
        print(f"  F1 Score: {qual['f1_mean']:.3f}")