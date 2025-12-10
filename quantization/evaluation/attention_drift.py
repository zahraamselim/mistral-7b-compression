"""
Attention Drift Benchmark - OPTIMIZED

Measures attention stability during generation in quantized models.

Optimizations:
- Pre-load and cache datasets with indexing
- Faster passage extraction
- Progress every 5 samples
- Better memory management
"""

import random
import numpy as np
import torch
import gc
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datasets import load_dataset
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


class AttentionDriftBenchmark:
    """
    Measures attention stability during text generation at document level.
    
    Metrics:
        - Mean drift: Average L1 distance between consecutive document attention distributions
        - Max drift: Maximum drift observed across generation
        - Drift by position: Drift at specific token positions
        - Drift from relevant doc: Specific drift from answer-containing document
        - Drift-quality correlation: Does drift predict answer quality?
    """
    
    @staticmethod
    def create_fast(model: ModelInterface):
        """Create FAST benchmark: 50 samples, 3 positions, 5 docs (~15-20 min)"""
        return AttentionDriftBenchmark(
            model=model,
            num_samples=50,
            generation_positions=[1, 5, 20],
            num_documents=5
        )
    
    @staticmethod
    def create_standard(model: ModelInterface):
        """Create STANDARD benchmark: 100 samples, 3 positions, 5 docs (~30-40 min)"""
        return AttentionDriftBenchmark(
            model=model,
            num_samples=100,
            generation_positions=[1, 5, 20],
            num_documents=5
        )
    
    @staticmethod
    def create_full(model: ModelInterface):
        """Create FULL benchmark: 150 samples, 5 positions, 5 docs (~60-80 min)"""
        return AttentionDriftBenchmark(
            model=model,
            num_samples=150,
            generation_positions=[1, 5, 10, 20, 40],
            num_documents=5
        )
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 100,
        generation_positions: List[int] = None,
        num_documents: int = 5,
        layers_to_analyze: Optional[List[int]] = None,
        random_seed: int = 42
    ):
        self.model = model
        self.num_samples = num_samples
        self.generation_positions = sorted(generation_positions or [1, 5, 20])
        self.num_documents = num_documents
        self.layers_to_analyze = layers_to_analyze
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.queries = None
        self.passages = None
        self.passage_index = None
    
    def _load_datasets(self) -> Tuple[any, List[Dict[str, str]]]:
        """Load Natural Questions dataset with Wikipedia corpus."""
        print("Loading datasets...")
        print("  Loading Natural Questions...")
        queries = load_dataset("nq_open", split="validation")
        print(f"    Loaded {len(queries)} questions")
        
        print("  Loading Wikipedia corpus...")
        try:
            wiki_dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True
            )
            
            passages = []
            for i, sample in enumerate(wiki_dataset):
                if i >= 500:
                    break
                
                text = sample.get('text', '')
                title = sample.get('title', '')
                if len(text) > 200:
                    passages.append({
                        'text': text,
                        'title': title,
                        'text_lower': text.lower()
                    })
                
                if i % 100 == 0 and i > 0:
                    print(f"    Loaded {i} documents...")
            
            if len(passages) < 50:
                raise RuntimeError("Insufficient Wikipedia passages loaded")
            
        except Exception as e:
            print(f"    Failed to load wikimedia: {e}")
            print("    Loading simple Wikipedia...")
            
            wiki_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:500]")
            passages = []
            for sample in wiki_dataset:
                text = sample.get('text', '')
                title = sample.get('title', '')
                if len(text) > 200:
                    passages.append({
                        'text': text,
                        'title': title,
                        'text_lower': text.lower()
                    })
        
        print(f"    Loaded {len(passages)} passages")
        
        # Build answer index
        print("  Building passage index...")
        passage_index = defaultdict(list)
        for idx, passage in enumerate(passages):
            words = passage['text_lower'].split()[:100]
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                passage_index[phrase].append(idx)
        
        print(f"    Indexed {len(passage_index)} phrases")
        print("Dataset loading complete!\n")
        
        return queries, passages, passage_index
    
    def _extract_relevant_passage_fast(
        self,
        answer: str
    ) -> Optional[str]:
        """Fast passage extraction using index."""
        answer_lower = answer.lower()
        
        # Try index first
        words = answer_lower.split()
        if len(words) >= 3:
            phrase = ' '.join(words[:3])
            if phrase in self.passage_index:
                candidates = self.passage_index[phrase]
                if candidates:
                    idx = random.choice(candidates[:10])
                    passage = self.passages[idx]
                    return self._extract_context(passage['text'], answer_lower)
        
        # Fallback to sampling
        sample_size = min(150, len(self.passages))
        for passage in random.sample(self.passages, sample_size):
            if answer_lower in passage['text_lower']:
                return self._extract_context(passage['text'], answer_lower)
        
        return None
    
    def _extract_context(self, text: str, answer_lower: str) -> Optional[str]:
        """Extract context around answer."""
        sentences = re.split(r'[.!?]+\s+', text)
        
        for i, sent in enumerate(sentences):
            if answer_lower in sent.lower():
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context = '. '.join(sentences[start:end])
                
                if len(context) > 80:
                    return context
        
        return None
    
    def _get_document_token_spans(
        self,
        full_tokens: torch.Tensor,
        doc_marker_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """Get exact token spans for each document using marker token IDs."""
        full_tokens_cpu = full_tokens[0].cpu().tolist()
        
        spans = []
        current_start = 0
        
        for i in range(self.num_documents):
            try:
                marker_positions = []
                for j in range(current_start, len(full_tokens_cpu)):
                    if full_tokens_cpu[j] in doc_marker_ids:
                        marker_positions.append(j)
                        if len(marker_positions) >= 2:
                            break
                
                if len(marker_positions) >= 2:
                    start = marker_positions[0]
                    end = marker_positions[1]
                    spans.append((start, end))
                    current_start = end
                elif len(marker_positions) == 1:
                    start = marker_positions[0]
                    end = len(full_tokens_cpu)
                    spans.append((start, end))
                    break
                else:
                    break
                    
            except Exception:
                break
        
        while len(spans) < self.num_documents:
            spans.append((0, 0))
        
        return spans[:self.num_documents]
    
    def _extract_document_attention(
        self,
        attentions: Tuple[torch.Tensor],
        input_ids: torch.Tensor,
        doc_marker_ids: List[int],
        num_generated: int
    ) -> Optional[np.ndarray]:
        """Extract document-level attention at a specific generation step."""
        doc_spans = self._get_document_token_spans(input_ids, doc_marker_ids)
        
        model_info = self.model.get_model_info()
        num_layers = model_info.get('num_layers', 32)
        
        if self.layers_to_analyze is None:
            start_layer = num_layers // 4
            end_layer = 3 * num_layers // 4
            layers_to_use = list(range(start_layer, end_layer))
        else:
            layers_to_use = self.layers_to_analyze
        
        doc_attention = np.zeros(self.num_documents)
        total_attention = 0.0
        
        if not attentions or len(attentions) == 0:
            return None
        
        for layer_idx in layers_to_use:
            if layer_idx >= len(attentions):
                continue
            
            try:
                layer_attn = attentions[layer_idx]
                avg_heads = layer_attn.mean(dim=1).squeeze()
                
                if avg_heads.dim() == 1:
                    token_attn = avg_heads.cpu().numpy()
                    for doc_idx, (start, end) in enumerate(doc_spans):
                        if start < end and end <= len(token_attn):
                            doc_attention[doc_idx] += token_attn[start:end].sum()
                            total_attention += token_attn[start:end].sum()
                else:
                    generated_start = max(0, avg_heads.shape[0] - num_generated)
                    
                    for token_idx in range(generated_start, avg_heads.shape[0]):
                        token_attn = avg_heads[token_idx, :].cpu().numpy()
                        
                        for doc_idx, (start, end) in enumerate(doc_spans):
                            if start < end and end <= len(token_attn):
                                doc_attention[doc_idx] += token_attn[start:end].sum()
                                total_attention += token_attn[start:end].sum()
                
                del layer_attn, avg_heads
                
            except Exception:
                continue
        
        if total_attention > 0:
            doc_attention = doc_attention / total_attention
        else:
            return None
        
        return doc_attention
    
    def _calculate_drift(
        self,
        attn_prev: np.ndarray,
        attn_curr: np.ndarray
    ) -> float:
        """Calculate L1 drift between two document attention distributions."""
        if len(attn_prev) != len(attn_curr):
            return 0.0
        
        if attn_prev.sum() > 0:
            attn_prev = attn_prev / attn_prev.sum()
        if attn_curr.sum() > 0:
            attn_curr = attn_curr / attn_curr.sum()
        
        drift = np.abs(attn_curr - attn_prev).sum()
        return float(drift)
    
    def _bootstrap_confidence_interval(
        self,
        data: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrapped_means = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrapped_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrapped_means, alpha/2 * 100)
        upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)
        
        return float(lower), float(upper)
    
    def run(self) -> Dict[str, any]:
        """Run attention drift benchmark."""
        print(f"Samples: {self.num_samples}, Positions: {self.generation_positions}, Documents: {self.num_documents}")
        
        self.queries, self.passages, self.passage_index = self._load_datasets()
        
        drift_scores = []
        max_drifts = []
        drift_by_position = defaultdict(list)
        drift_from_relevant = []
        
        f1_scores = []
        drift_when_correct = []
        drift_when_wrong = []
        relevant_drift_when_correct = []
        relevant_drift_when_wrong = []
        
        doc_marker_ids = [self.model.tokenizer.encode("\n\n", add_special_tokens=False)[0]]
        
        max_pos = max(self.generation_positions)
        config = GenerationConfig(
            max_new_tokens=max_pos,
            do_sample=False,
            temperature=1.0
        )
        
        successful = 0
        attempted = 0
        skipped = 0
        start_time = time.time()
        
        print(f"\nStarting evaluation loop (target: {self.num_samples} samples)...")
        print("Progress: [successful/target] (attempted) - metrics")
        
        while successful < self.num_samples and attempted < len(self.queries):
            sample = self.queries[attempted]
            attempted += 1
            
            query = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not query or not answers:
                continue
            
            answer = answers[0]
            
            relevant_passage = self._extract_relevant_passage_fast(answer)
            if not relevant_passage:
                skipped += 1
                continue
            
            distractor_passages = []
            for passage in random.sample(self.passages, min(80, len(self.passages))):
                text = passage['text'][:300]
                if (answer.lower() not in text.lower() and 
                    text != relevant_passage and
                    len(distractor_passages) < self.num_documents - 1):
                    distractor_passages.append(text)
            
            if len(distractor_passages) < self.num_documents - 1:
                skipped += 1
                continue
            
            all_docs = [relevant_passage[:400]] + [d[:400] for d in distractor_passages]
            random.shuffle(all_docs)
            relevant_idx = all_docs.index(relevant_passage[:400])
            
            docs_text = "\n\n".join([
                f"Document {j+1}: {doc}" 
                for j, doc in enumerate(all_docs)
            ])
            
            prompt = f"Question: {query}\n\n{docs_text}\n\nAnswer:"
            
            try:
                sample_start = time.time()
                
                inputs = self.model.encode(prompt, max_length=2048)
                output = self.model.generate(prompt, config, return_attentions=True)
                
                sample_time = time.time() - sample_start
                
                generated_text = output.generated_text
                f1 = max([compute_f1(generated_text, ans) for ans in answers])
                f1_scores.append(f1)
                
                is_correct = (f1 > 0.3)
                
                if not output.attentions or len(output.attentions) < min(self.generation_positions):
                    continue
                
                attention_sequence = []
                
                for pos_idx, pos in enumerate(self.generation_positions):
                    if pos > len(output.attentions):
                        break
                    
                    step_attentions = output.attentions[pos - 1]
                    
                    doc_attn = self._extract_document_attention(
                        step_attentions,
                        inputs['input_ids'],
                        doc_marker_ids,
                        pos
                    )
                    
                    if doc_attn is not None:
                        attention_sequence.append(doc_attn)
                    
                    del step_attentions
                    gc.collect()
                
                if len(attention_sequence) > 1:
                    position_drifts = []
                    
                    for j in range(1, len(attention_sequence)):
                        drift = self._calculate_drift(
                            attention_sequence[j-1],
                            attention_sequence[j]
                        )
                        position_drifts.append(drift)
                        
                        if j < len(self.generation_positions):
                            drift_by_position[self.generation_positions[j]].append(drift)
                        
                        relevant_drift = abs(
                            attention_sequence[j][relevant_idx] - 
                            attention_sequence[j-1][relevant_idx]
                        )
                        drift_from_relevant.append(float(relevant_drift))
                        
                        if is_correct:
                            relevant_drift_when_correct.append(float(relevant_drift))
                        else:
                            relevant_drift_when_wrong.append(float(relevant_drift))
                    
                    mean_drift = float(np.mean(position_drifts))
                    max_drift = float(np.max(position_drifts))
                    
                    drift_scores.append(mean_drift)
                    max_drifts.append(max_drift)
                    
                    if is_correct:
                        drift_when_correct.append(mean_drift)
                    else:
                        drift_when_wrong.append(mean_drift)
                    
                    successful += 1
                    
                    if successful % 5 == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / successful
                        remaining = (self.num_samples - successful) * avg_time
                        current_drift = np.mean(drift_scores)
                        current_f1 = np.mean(f1_scores)
                        
                        print(f"[{successful}/{self.num_samples}] ({attempted}) - "
                              f"Drift={current_drift:.4f} F1={current_f1:.3f} "
                              f"AvgTime={sample_time:.1f}s ETA={remaining/60:.1f}min "
                              f"Skip={skipped}")
                
                del inputs, output, attention_sequence
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[{successful}/{self.num_samples}] OOM, clearing memory...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    continue
            except Exception as e:
                if successful % 10 == 0:
                    print(f"[{successful}/{self.num_samples}] Error: {type(e).__name__}")
                continue
        
        if len(drift_scores) == 0:
            raise ValueError("No successful evaluations completed")
        
        mean_drift = float(np.mean(drift_scores))
        mean_drift_ci = self._bootstrap_confidence_interval(drift_scores)
        
        max_drift_mean = float(np.mean(max_drifts))
        max_drift_std = float(np.std(max_drifts))
        
        relevant_drift_mean = float(np.mean(drift_from_relevant)) if drift_from_relevant else 0.0
        relevant_drift_std = float(np.std(drift_from_relevant)) if drift_from_relevant else 0.0
        
        drift_by_pos_stats = {}
        for pos in self.generation_positions[1:]:
            if pos in drift_by_position and len(drift_by_position[pos]) > 0:
                drift_by_pos_stats[str(pos)] = {
                    "mean": float(np.mean(drift_by_position[pos])),
                    "std": float(np.std(drift_by_position[pos])),
                    "median": float(np.median(drift_by_position[pos])),
                    "n": len(drift_by_position[pos])
                }
        
        from scipy.stats import pearsonr, mannwhitneyu
        
        drift_quality_correlation = {}
        if len(drift_scores) == len(f1_scores):
            pearson_r, pearson_p = pearsonr(drift_scores, f1_scores)
            
            drift_quality_correlation = {
                "drift_vs_f1_pearson_r": float(pearson_r),
                "drift_vs_f1_pearson_p": float(pearson_p),
                "interpretation": (
                    "high_drift_predicts_low_quality" if pearson_p < 0.05 and pearson_r < -0.3
                    else "high_drift_predicts_high_quality" if pearson_p < 0.05 and pearson_r > 0.3
                    else "no_significant_correlation"
                )
            }
        
        drift_by_correctness = {
            "drift_when_correct_mean": float(np.mean(drift_when_correct)) if drift_when_correct else 0.0,
            "drift_when_correct_std": float(np.std(drift_when_correct)) if drift_when_correct else 0.0,
            "drift_when_wrong_mean": float(np.mean(drift_when_wrong)) if drift_when_wrong else 0.0,
            "drift_when_wrong_std": float(np.std(drift_when_wrong)) if drift_when_wrong else 0.0,
            "relevant_drift_correct_mean": float(np.mean(relevant_drift_when_correct)) if relevant_drift_when_correct else 0.0,
            "relevant_drift_wrong_mean": float(np.mean(relevant_drift_when_wrong)) if relevant_drift_when_wrong else 0.0,
        }
        
        if len(drift_when_correct) > 5 and len(drift_when_wrong) > 5:
            u_stat, p_val = mannwhitneyu(drift_when_correct, drift_when_wrong, alternative='two-sided')
            drift_by_correctness["statistical_test"] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "significant_difference": p_val < 0.05
            }
        
        results = {
            "mean_drift": mean_drift,
            "mean_drift_ci_95_lower": mean_drift_ci[0],
            "mean_drift_ci_95_upper": mean_drift_ci[1],
            "max_drift_mean": max_drift_mean,
            "max_drift_std": max_drift_std,
            "drift_from_relevant_mean": relevant_drift_mean,
            "drift_from_relevant_std": relevant_drift_std,
            "drift_by_position": drift_by_pos_stats,
            "drift_quality_correlation": drift_quality_correlation,
            "drift_by_correctness": drift_by_correctness,
            "num_samples": len(drift_scores),
            "samples_attempted": attempted,
            "samples_skipped": skipped,
            "generation_positions": self.generation_positions,
            "methodology": {
                "drift_measurement": "document_level_not_token_level",
                "layers_analyzed": "middle_50_percent" if self.layers_to_analyze is None else self.layers_to_analyze,
                "attention_aggregation": "per_step_no_double_counting",
                "answer_quality_measured": True,
                "optimizations": "indexed_passages_cached_data"
            }
        }
        
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time/60:.1f} minutes")
        print(f"Mean drift: {mean_drift:.4f} [{mean_drift_ci[0]:.4f}, {mean_drift_ci[1]:.4f}]")
        print(f"Drift when correct: {drift_by_correctness['drift_when_correct_mean']:.4f}, when wrong: {drift_by_correctness['drift_when_wrong_mean']:.4f}")
        print(f"Drift-quality correlation: r={drift_quality_correlation.get('drift_vs_f1_pearson_r', 0):.3f}")
        print(f"Success rate: {successful}/{attempted} ({100*successful/max(attempted, 1):.1f}%)")
        
        return results