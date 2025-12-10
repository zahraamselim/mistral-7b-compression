"""
Attention Preservation Benchmark - OPTIMIZED

Measures whether quantized models preserve attention to relevant documents in 
multi-document retrieval scenarios.

Optimizations:
- Pre-load and cache datasets
- Batch tokenization
- Pre-compute BM25 index
- Faster passage matching with indexing
- Reduced dataset loading overhead
"""

import random
import numpy as np
import torch
import gc
import time
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

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


def compute_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction exactly matches any ground truth."""
    pred_normalized = prediction.lower().strip()
    
    for gt in ground_truths:
        gt_normalized = gt.lower().strip()
        if pred_normalized == gt_normalized or gt_normalized in pred_normalized:
            return True
    
    return False


class DatasetCache:
    """Cache for pre-loaded and indexed datasets."""
    
    def __init__(self):
        self.nq_dataset = None
        self.wiki_passages = None
        self.answer_to_passages = None
        self.bm25_index = None
        self.bm25_passages = None
        
    def load(self, max_wiki_docs: int = 1000):
        """Load and index all datasets once."""
        print("Loading and indexing datasets (one-time setup)...")
        
        # Load Natural Questions
        print("  Loading Natural Questions...")
        self.nq_dataset = load_dataset("nq_open", split="validation")
        print(f"    Loaded {len(self.nq_dataset)} questions")
        
        # Load Wikipedia
        print("  Loading Wikipedia corpus...")
        try:
            wiki_dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True
            )
            
            self.wiki_passages = []
            for i, sample in enumerate(wiki_dataset):
                if i >= max_wiki_docs:
                    break
                
                text = sample.get('text', '')
                title = sample.get('title', '')
                if len(text) > 200:
                    self.wiki_passages.append({
                        'text': text,
                        'title': title,
                        'text_lower': text.lower()
                    })
                
                if i % 200 == 0 and i > 0:
                    print(f"    Loaded {i} documents...")
            
        except Exception as e:
            print(f"    Failed to load wikimedia: {e}")
            print("    Loading simple Wikipedia...")
            
            wiki_dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{max_wiki_docs}]")
            self.wiki_passages = []
            for sample in wiki_dataset:
                text = sample.get('text', '')
                title = sample.get('title', '')
                if len(text) > 200:
                    self.wiki_passages.append({
                        'text': text,
                        'title': title,
                        'text_lower': text.lower()
                    })
        
        print(f"    Loaded {len(self.wiki_passages)} passages")
        
        # Build answer index
        print("  Building answer index...")
        self.answer_to_passages = defaultdict(list)
        
        for idx, passage in enumerate(self.wiki_passages):
            # Extract key phrases (simple approach: first 100 words)
            words = passage['text_lower'].split()[:100]
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                self.answer_to_passages[phrase].append(idx)
        
        print(f"    Indexed {len(self.answer_to_passages)} phrases")
        
        # Build BM25 index
        print("  Building BM25 index...")
        self.bm25_passages = [p['text'][:800] for p in self.wiki_passages]
        tokenized_corpus = [doc.lower().split()[:100] for doc in self.bm25_passages]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print("    BM25 index built")
        
        print("Dataset loading complete!\n")


class AttentionPreservationBenchmark:
    """
    Measures attention preservation to relevant documents in quantized models.
    
    Metrics:
        - Attention Precision@1: Fraction where highest attention goes to answer doc
        - Attention Rank: Rank of answer document in attention distribution
        - Attention Concentration: Gini coefficient measuring attention spread
        - Answer Accuracy: Exact match and F1 scores
        - Attention-Accuracy Correlation: Does attention predict correctness?
    """
    
    @staticmethod
    def create_fast(model: ModelInterface):
        """Create FAST benchmark: 50 samples, 5 docs (~10-15 min)"""
        return AttentionPreservationBenchmark(
            model=model,
            num_samples=50,
            num_documents=5,
            max_doc_tokens=128,
            test_aggregation_strategies=False
        )
    
    @staticmethod
    def create_standard(model: ModelInterface):
        """Create STANDARD benchmark: 100 samples, 5 docs (~20-30 min)"""
        return AttentionPreservationBenchmark(
            model=model,
            num_samples=100,
            num_documents=5,
            max_doc_tokens=128,
            test_aggregation_strategies=False
        )
    
    @staticmethod
    def create_full(model: ModelInterface):
        """Create FULL benchmark: 300 samples, 10 docs (~60-90 min)"""
        return AttentionPreservationBenchmark(
            model=model,
            num_samples=300,
            num_documents=10,
            max_doc_tokens=128,
            test_aggregation_strategies=True
        )
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 100,
        num_documents: int = 5,
        max_doc_tokens: int = 128,
        layers_to_analyze: Optional[List[int]] = None,
        test_aggregation_strategies: bool = False,
        random_seed: int = 42
    ):
        self.model = model
        self.num_samples = num_samples
        self.num_documents = num_documents
        self.max_doc_tokens = max_doc_tokens
        self.layers_to_analyze = layers_to_analyze
        self.test_aggregation_strategies = test_aggregation_strategies
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.dataset_cache = DatasetCache()
        self.power_analysis = self._calculate_statistical_power()
    
    def _calculate_statistical_power(self) -> Dict[str, float]:
        """Calculate statistical power for the given sample size."""
        effect_size = 0.3
        alpha = 0.05
        n = self.num_samples
        
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = (effect_size * np.sqrt(n)) - z_alpha
        power = norm.cdf(z_beta)
        
        return {
            "effect_size": effect_size,
            "alpha": alpha,
            "n_samples": n,
            "actual_power": float(power),
            "adequate": power >= 0.80
        }
    
    def _extract_answer_context_fast(
        self, 
        answer: str
    ) -> Optional[str]:
        """Fast passage lookup using pre-built index."""
        answer_lower = answer.lower()
        
        # Try exact phrase matches first
        words = answer_lower.split()
        if len(words) >= 3:
            phrase = ' '.join(words[:3])
            if phrase in self.dataset_cache.answer_to_passages:
                candidates = self.dataset_cache.answer_to_passages[phrase]
                if candidates:
                    idx = random.choice(candidates[:10])
                    passage = self.dataset_cache.wiki_passages[idx]
                    return self._extract_context_from_passage(passage['text'], answer_lower)
        
        # Fallback to linear search on subset
        sample_size = min(200, len(self.dataset_cache.wiki_passages))
        for passage in random.sample(self.dataset_cache.wiki_passages, sample_size):
            if answer_lower in passage['text_lower']:
                return self._extract_context_from_passage(passage['text'], answer_lower)
        
        return None
    
    def _extract_context_from_passage(self, text: str, answer_lower: str) -> Optional[str]:
        """Extract context around answer."""
        sentences = re.split(r'[.!?]+', text)
        
        for i, sent in enumerate(sentences):
            if answer_lower in sent.lower():
                start_idx = max(0, i - 1)
                end_idx = min(len(sentences), i + 2)
                context = '. '.join(sentences[start_idx:end_idx])
                
                if len(context) > 50:
                    return context
        
        return None
    
    def _select_distractors_bm25_fast(
        self,
        query: str,
        answer: str,
        relevant_doc: str,
        num_distractors: int
    ) -> List[str]:
        """Fast BM25 distractor selection using pre-built index."""
        answer_lower = answer.lower()
        
        # Get BM25 scores
        query_tokens = query.lower().split()[:20]
        scores = self.dataset_cache.bm25_index.get_scores(query_tokens)
        
        # Get top candidates excluding answer-containing docs
        top_indices = np.argsort(scores)[::-1]
        
        selected = []
        for idx in top_indices:
            if len(selected) >= num_distractors:
                break
            
            doc_text = self.dataset_cache.bm25_passages[idx]
            
            # Skip if contains answer or is the relevant doc
            if answer_lower in doc_text.lower() or doc_text == relevant_doc:
                continue
            
            # Extract context
            sentences = re.split(r'[.!?]+', doc_text)
            if len(sentences) > 2:
                context = '. '.join(sentences[:3])
                selected.append(context)
        
        # Fill remaining with random
        while len(selected) < num_distractors:
            idx = random.randint(0, len(self.dataset_cache.wiki_passages) - 1)
            text = self.dataset_cache.wiki_passages[idx]['text'][:400]
            if text not in selected and answer_lower not in text.lower():
                selected.append(text)
        
        return selected[:num_distractors]
    
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
    
    def _aggregate_attention_to_documents(
        self,
        attentions: Tuple[torch.Tensor],
        input_ids: torch.Tensor,
        doc_marker_ids: List[int],
        num_generated: int,
        aggregation_strategy: str = "middle_50"
    ) -> np.ndarray:
        """Aggregate attention weights to document level."""
        doc_spans = self._get_document_token_spans(input_ids, doc_marker_ids)
        
        model_info = self.model.get_model_info()
        num_layers = model_info.get('num_layers', 32)
        
        if aggregation_strategy == "middle_50":
            start_layer = num_layers // 4
            end_layer = 3 * num_layers // 4
            layers_to_use = list(range(start_layer, end_layer))
        elif aggregation_strategy == "all_layers":
            layers_to_use = list(range(num_layers))
        elif aggregation_strategy == "last_layer":
            layers_to_use = [num_layers - 1]
        else:
            layers_to_use = self.layers_to_analyze or list(range(num_layers // 4, 3 * num_layers // 4))
        
        doc_attention = np.zeros(self.num_documents)
        total_attention = 0.0
        
        if not attentions or len(attentions) == 0:
            return doc_attention
        
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
        
        return doc_attention
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for attention concentration."""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        total = cumsum[-1]
        
        if total == 0:
            return 0.0
        
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_values)) / (n * total) - (n + 1) / n
        return float(gini)
    
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
        """Run attention preservation benchmark."""
        print(f"Samples: {self.num_samples}, Documents: {self.num_documents}, Power: {self.power_analysis['actual_power']:.3f}")
        
        # Load and index datasets once
        self.dataset_cache.load(max_wiki_docs=1000)
        
        precision_at_1 = []
        attention_ranks = []
        attention_gini = []
        
        exact_matches = []
        f1_scores = []
        
        correct_with_high_attention = []
        correct_with_low_attention = []
        attention_scores_when_correct = []
        attention_scores_when_wrong = []
        
        aggregation_results = {
            "middle_50": {"precision": [], "rank": []},
            "all_layers": {"precision": [], "rank": []},
            "last_layer": {"precision": [], "rank": []}
        } if self.test_aggregation_strategies else None
        
        config = GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0
        )
        
        doc_marker_ids = [self.model.tokenizer.encode("\n\n", add_special_tokens=False)[0]]
        
        successful = 0
        attempted = 0
        skipped_no_context = 0
        skipped_no_distractors = 0
        start_time = time.time()
        
        print(f"\nStarting evaluation loop (target: {self.num_samples} samples)...")
        print("Progress: [successful/target] (attempted) - metrics")
        
        while successful < self.num_samples and attempted < len(self.dataset_cache.nq_dataset):
            sample = self.dataset_cache.nq_dataset[attempted]
            attempted += 1
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not answers or not question:
                continue
            
            answer = answers[0]
            
            # Fast context extraction
            relevant_doc = self._extract_answer_context_fast(answer)
            if not relevant_doc:
                skipped_no_context += 1
                continue
            
            # Fast distractor selection
            distractor_docs = self._select_distractors_bm25_fast(
                question,
                answer,
                relevant_doc,
                self.num_documents - 1
            )
            
            if len(distractor_docs) < self.num_documents - 1:
                skipped_no_distractors += 1
                continue
            
            all_docs = [relevant_doc[:600]] + [d[:600] for d in distractor_docs]
            random.shuffle(all_docs)
            answer_idx = all_docs.index(relevant_doc[:600])
            
            docs_text = "\n\n".join([
                f"Document {j+1}: {doc}" for j, doc in enumerate(all_docs)
            ])
            
            prompt = (
                f"Question: {question}\n\n"
                f"{docs_text}\n\n"
                f"Answer:"
            )
            
            try:
                sample_start = time.time()
                
                inputs = self.model.encode(prompt, max_length=2048)
                output = self.model.generate(prompt, config, return_attentions=True)
                
                sample_time = time.time() - sample_start
                
                generated_text = output.generated_text
                
                em = compute_exact_match(generated_text, answers)
                f1 = max([compute_f1(generated_text, ans) for ans in answers])
                
                exact_matches.append(1.0 if em else 0.0)
                f1_scores.append(f1)
                
                if output.attentions and len(output.attentions) > 0:
                    final_step_attentions = output.attentions[-1]
                    
                    doc_attention = self._aggregate_attention_to_documents(
                        final_step_attentions,
                        inputs['input_ids'],
                        doc_marker_ids,
                        output.num_generated_tokens,
                        aggregation_strategy="middle_50"
                    )
                    
                    if doc_attention.sum() > 0:
                        top_doc = np.argmax(doc_attention)
                        high_attention = (top_doc == answer_idx)
                        
                        precision_at_1.append(1.0 if high_attention else 0.0)
                        
                        sorted_indices = np.argsort(doc_attention)[::-1]
                        rank = int(np.where(sorted_indices == answer_idx)[0][0] + 1)
                        attention_ranks.append(rank)
                        
                        gini = self._calculate_gini_coefficient(doc_attention)
                        attention_gini.append(gini)
                        
                        relevant_doc_attention = doc_attention[answer_idx]
                        
                        if em or f1 > 0.3:
                            attention_scores_when_correct.append(relevant_doc_attention)
                            if high_attention:
                                correct_with_high_attention.append(1.0)
                            else:
                                correct_with_low_attention.append(1.0)
                        else:
                            attention_scores_when_wrong.append(relevant_doc_attention)
                        
                        if self.test_aggregation_strategies:
                            for strategy in ["all_layers", "last_layer"]:
                                doc_attn_alt = self._aggregate_attention_to_documents(
                                    final_step_attentions,
                                    inputs['input_ids'],
                                    doc_marker_ids,
                                    output.num_generated_tokens,
                                    aggregation_strategy=strategy
                                )
                                
                                if doc_attn_alt.sum() > 0:
                                    top_doc_alt = np.argmax(doc_attn_alt)
                                    aggregation_results[strategy]["precision"].append(
                                        1.0 if top_doc_alt == answer_idx else 0.0
                                    )
                                    
                                    sorted_indices_alt = np.argsort(doc_attn_alt)[::-1]
                                    rank_alt = int(np.where(sorted_indices_alt == answer_idx)[0][0] + 1)
                                    aggregation_results[strategy]["rank"].append(rank_alt)
                        
                        successful += 1
                        
                        if successful % 5 == 0:
                            elapsed = time.time() - start_time
                            avg_time = elapsed / successful
                            remaining = (self.num_samples - successful) * avg_time
                            current_precision = np.mean(precision_at_1)
                            current_f1 = np.mean(f1_scores)
                            
                            print(f"[{successful}/{self.num_samples}] ({attempted}) - "
                                  f"Prec@1={current_precision:.3f} F1={current_f1:.3f} "
                                  f"AvgTime={sample_time:.1f}s ETA={remaining/60:.1f}min "
                                  f"Skip:ctx={skipped_no_context},dist={skipped_no_distractors}")
                
                del inputs, output
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
        
        if len(precision_at_1) == 0:
            raise ValueError("No successful evaluations completed")
        
        precision_mean = float(np.mean(precision_at_1))
        precision_ci = self._bootstrap_confidence_interval(precision_at_1)
        
        rank_mean = float(np.mean(attention_ranks))
        rank_median = float(np.median(attention_ranks))
        rank_std = float(np.std(attention_ranks))
        
        gini_mean = float(np.mean(attention_gini))
        gini_std = float(np.std(attention_gini))
        
        em_mean = float(np.mean(exact_matches))
        f1_mean = float(np.mean(f1_scores))
        
        from scipy.stats import pearsonr, spearmanr
        
        attention_quality_correlation = {}
        if len(precision_at_1) == len(f1_scores):
            pearson_r, pearson_p = pearsonr(precision_at_1, f1_scores)
            spearman_r, spearman_p = spearmanr(attention_ranks, f1_scores)
            
            attention_quality_correlation = {
                "precision_vs_f1_pearson_r": float(pearson_r),
                "precision_vs_f1_pearson_p": float(pearson_p),
                "rank_vs_f1_spearman_r": float(spearman_r),
                "rank_vs_f1_spearman_p": float(spearman_p),
                "interpretation": (
                    "significant_positive" if pearson_p < 0.05 and pearson_r > 0
                    else "significant_negative" if pearson_p < 0.05 and pearson_r < 0
                    else "no_significant_correlation"
                )
            }
        
        attention_by_correctness = {
            "attention_when_correct_mean": float(np.mean(attention_scores_when_correct)) if attention_scores_when_correct else 0.0,
            "attention_when_correct_std": float(np.std(attention_scores_when_correct)) if attention_scores_when_correct else 0.0,
            "attention_when_wrong_mean": float(np.mean(attention_scores_when_wrong)) if attention_scores_when_wrong else 0.0,
            "attention_when_wrong_std": float(np.std(attention_scores_when_wrong)) if attention_scores_when_wrong else 0.0,
            "correct_with_high_attention": len(correct_with_high_attention),
            "correct_with_low_attention": len(correct_with_low_attention),
            "high_attention_improves_accuracy": (
                len(correct_with_high_attention) > len(correct_with_low_attention)
            )
        }
        
        results = {
            "attention_precision_at_1": precision_mean,
            "attention_precision_at_1_ci_95_lower": precision_ci[0],
            "attention_precision_at_1_ci_95_upper": precision_ci[1],
            "attention_rank_mean": rank_mean,
            "attention_rank_median": rank_median,
            "attention_rank_std": rank_std,
            "attention_concentration_gini": gini_mean,
            "attention_concentration_gini_std": gini_std,
            "exact_match_accuracy": em_mean,
            "f1_score_mean": f1_mean,
            "attention_quality_correlation": attention_quality_correlation,
            "attention_by_correctness": attention_by_correctness,
            "num_samples": len(precision_at_1),
            "samples_attempted": attempted,
            "skipped_no_context": skipped_no_context,
            "skipped_no_distractors": skipped_no_distractors,
            "power_analysis": self.power_analysis,
            "methodology": {
                "layers_analyzed": "middle_50_percent" if self.layers_to_analyze is None else self.layers_to_analyze,
                "attention_aggregation": "final_step_only_no_double_counting",
                "distractor_selection": "bm25_semantic",
                "answer_quality_measured": True,
                "optimizations": "pre_indexed_datasets_cached_bm25"
            }
        }
        
        if self.test_aggregation_strategies and aggregation_results:
            results["aggregation_comparison"] = {
                strategy: {
                    "precision_at_1": float(np.mean(data["precision"])) if data["precision"] else 0.0,
                    "rank_mean": float(np.mean(data["rank"])) if data["rank"] else 0.0
                }
                for strategy, data in aggregation_results.items()
            }
        
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time/60:.1f} minutes")
        print(f"Precision@1: {precision_mean:.3f} [{precision_ci[0]:.3f}, {precision_ci[1]:.3f}]")
        print(f"Mean rank: {rank_mean:.2f}, Gini: {gini_mean:.3f}")
        print(f"Answer quality: EM={em_mean:.3f}, F1={f1_mean:.3f}")
        print(f"Attention-quality correlation: r={attention_quality_correlation.get('precision_vs_f1_pearson_r', 0):.3f}")
        print(f"Success rate: {successful}/{attempted} ({100*successful/max(attempted, 1):.1f}%)")
        
        return results