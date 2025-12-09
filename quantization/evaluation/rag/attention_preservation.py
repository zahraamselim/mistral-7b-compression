"""
Attention Preservation Benchmark

Measures whether quantized models preserve attention to relevant documents in 
multi-document retrieval scenarios.

Research Question:
    Does quantization degrade the model's ability to identify and attend to 
    relevant information in a multi-document context?

Methodology:
    - Present model with N documents (1 relevant + N-1 distractors)
    - Track actual attention weights during generation
    - Aggregate attention properly: final generation step only, middle layers
    - Measure if highest attention goes to relevant document
    - Calculate attention rank and concentration metrics

Fixed for Kaggle T4 (15GB VRAM):
    - Proper attention aggregation (no double-counting)
    - Memory-efficient processing (delete tensors immediately)
    - Works with 4-bit quantized Mistral-7B
    - Batch size 1, gradient checkpointing compatible
"""

import random
import numpy as np
import torch
import gc
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from rank_bm25 import BM25Okapi
import re

from models.model_interface import ModelInterface, GenerationConfig


class AttentionPreservationBenchmark:
    """
    Measures attention preservation to relevant documents in quantized models.
    
    Metrics:
        - Attention Precision@1: Fraction of cases where highest attention 
          goes to the answer-containing document
        - Attention Rank: Rank of answer document in attention distribution
        - Attention Concentration: Gini coefficient measuring attention spread
    """
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 300,
        num_documents: int = 10,
        max_doc_tokens: int = 128,
        layers_to_analyze: Optional[List[int]] = None,
        random_seed: int = 42
    ):
        """
        Initialize attention preservation benchmark.
        
        Args:
            model: Model interface instance
            num_samples: Number of test samples
            num_documents: Number of documents per test (1 relevant + rest distractors)
            max_doc_tokens: Maximum tokens per document (reduced for T4)
            layers_to_analyze: Which layers to analyze (None = middle 50%)
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.num_samples = num_samples
        self.num_documents = num_documents
        self.max_doc_tokens = max_doc_tokens
        self.layers_to_analyze = layers_to_analyze
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
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
    
    def _load_datasets(self) -> Tuple[any, List[Dict[str, str]]]:
        """Load Natural Questions dataset with contexts."""
        print("Loading Natural Questions dataset with contexts...")
        
        nq_dataset = load_dataset("nq_open", split="validation")
        
        print("Loading Wikipedia corpus...")
        wiki_dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train[:2000]"
        )
        
        wiki_samples = []
        for sample in wiki_dataset:
            text = sample.get('text', '')
            title = sample.get('title', '')
            if len(text) > 200:
                wiki_samples.append({
                    'text': text,
                    'title': title
                })
        
        print(f"Loaded {len(wiki_samples)} Wikipedia documents")
        return nq_dataset, wiki_samples
    
    def _extract_answer_context(
        self, 
        answer: str, 
        wiki_samples: List[Dict[str, str]]
    ) -> Optional[str]:
        """Find Wikipedia passage that naturally contains the answer."""
        answer_lower = answer.lower()
        
        for sample in random.sample(wiki_samples, min(100, len(wiki_samples))):
            text = sample['text']
            if answer_lower in text.lower():
                sentences = re.split(r'[.!?]+', text)
                
                for i, sent in enumerate(sentences):
                    if answer_lower in sent.lower():
                        start_idx = max(0, i - 1)
                        end_idx = min(len(sentences), i + 2)
                        context = '. '.join(sentences[start_idx:end_idx])
                        
                        if len(context) > 50:
                            return context
        
        return None
    
    def _select_distractors_bm25(
        self,
        query: str,
        answer: str,
        relevant_doc: str,
        wiki_samples: List[Dict[str, str]],
        num_distractors: int
    ) -> List[str]:
        """Select plausible distractors using BM25."""
        candidate_docs = []
        candidate_texts = []
        
        answer_lower = answer.lower()
        
        for sample in wiki_samples[:500]:
            text = sample['text'][:800]
            if (answer_lower not in text.lower() and 
                text != relevant_doc and
                len(text) > 100):
                candidate_docs.append(sample)
                candidate_texts.append(text)
        
        if len(candidate_docs) < num_distractors:
            return [doc['text'][:400] for doc in random.sample(candidate_docs, min(len(candidate_docs), num_distractors))]
        
        tokenized_corpus = [doc.lower().split()[:100] for doc in candidate_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        
        query_tokens = query.lower().split()[:20]
        scores = bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:num_distractors * 2]
        
        selected = []
        for idx in top_indices:
            if len(selected) >= num_distractors:
                break
            doc_text = candidate_docs[idx]['text']
            
            sentences = re.split(r'[.!?]+', doc_text)
            if len(sentences) > 2:
                context = '. '.join(sentences[:3])
                selected.append(context)
        
        while len(selected) < num_distractors and candidate_docs:
            doc = random.choice(candidate_docs)
            text = doc['text'][:400]
            if text not in selected:
                selected.append(text)
        
        return selected[:num_distractors]
    
    def _get_document_token_spans(
        self,
        full_tokens: torch.Tensor,
        doc_marker_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Get exact token spans for each document using marker token IDs.
        
        For Mistral, we use newline tokens as document separators.
        """
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
        num_generated: int
    ) -> np.ndarray:
        """
        Aggregate attention weights to document level.
        
        CRITICAL FIX: Uses only final generation step, not all steps.
        This avoids double-counting early tokens.
        
        Args:
            attentions: Tuple of attention tensors from FINAL generation step
            input_ids: Input token IDs [1, seq_len]
            doc_marker_ids: Token IDs that mark document boundaries
            num_generated: Number of generated tokens
            
        Returns:
            Array of per-document attention scores [num_docs]
        """
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
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: Failed processing layer {layer_idx}: {str(e)}")
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
        print("\nAttention Preservation Benchmark")
        print(f"Samples: {self.num_samples}")
        print(f"Documents per sample: {self.num_documents}")
        print(f"Statistical power: {self.power_analysis['actual_power']:.3f}")
        
        nq_dataset, wiki_samples = self._load_datasets()
        
        precision_at_1 = []
        attention_ranks = []
        attention_gini = []
        
        config = GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0
        )
        
        print(f"\nProcessing {self.num_samples} samples...")
        
        doc_marker_ids = [self.model.tokenizer.encode("\n\n", add_special_tokens=False)[0]]
        
        successful = 0
        attempted = 0
        
        while successful < self.num_samples and attempted < len(nq_dataset):
            sample = nq_dataset[attempted]
            attempted += 1
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not answers or not question:
                continue
            
            answer = answers[0]
            
            relevant_doc = self._extract_answer_context(answer, wiki_samples)
            if not relevant_doc:
                continue
            
            distractor_docs = self._select_distractors_bm25(
                question,
                answer,
                relevant_doc,
                wiki_samples,
                self.num_documents - 1
            )
            
            if len(distractor_docs) < self.num_documents - 1:
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
                inputs = self.model.encode(prompt, max_length=2048)
                output = self.model.generate(prompt, config, return_attentions=True)
                
                if output.attentions and len(output.attentions) > 0:
                    final_step_attentions = output.attentions[-1]
                    
                    doc_attention = self._aggregate_attention_to_documents(
                        final_step_attentions,
                        inputs['input_ids'],
                        doc_marker_ids,
                        output.num_generated_tokens
                    )
                    
                    if doc_attention.sum() > 0:
                        top_doc = np.argmax(doc_attention)
                        precision_at_1.append(1.0 if top_doc == answer_idx else 0.0)
                        
                        sorted_indices = np.argsort(doc_attention)[::-1]
                        rank = int(np.where(sorted_indices == answer_idx)[0][0] + 1)
                        attention_ranks.append(rank)
                        
                        gini = self._calculate_gini_coefficient(doc_attention)
                        attention_gini.append(gini)
                        
                        successful += 1
                
                del inputs, output
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM on sample {attempted}, skipping...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Warning: Failed on sample {attempted}: {str(e)}")
                    continue
            except Exception as e:
                print(f"Warning: Failed on sample {attempted}: {str(e)}")
                continue
            
            if successful % 50 == 0 and successful > 0:
                print(f"Progress: {successful}/{self.num_samples}")
        
        if len(precision_at_1) == 0:
            raise ValueError("No successful evaluations completed")
        
        precision_mean = float(np.mean(precision_at_1))
        precision_ci = self._bootstrap_confidence_interval(precision_at_1)
        
        rank_mean = float(np.mean(attention_ranks))
        rank_median = float(np.median(attention_ranks))
        rank_std = float(np.std(attention_ranks))
        
        gini_mean = float(np.mean(attention_gini))
        gini_std = float(np.std(attention_gini))
        
        results = {
            "attention_precision_at_1": precision_mean,
            "attention_precision_at_1_ci_95_lower": precision_ci[0],
            "attention_precision_at_1_ci_95_upper": precision_ci[1],
            "attention_rank_mean": rank_mean,
            "attention_rank_median": rank_median,
            "attention_rank_std": rank_std,
            "attention_concentration_gini": gini_mean,
            "attention_concentration_gini_std": gini_std,
            "num_samples": len(precision_at_1),
            "samples_attempted": attempted,
            "power_analysis": self.power_analysis,
            "methodology": {
                "layers_analyzed": "middle_50_percent" if self.layers_to_analyze is None else self.layers_to_analyze,
                "attention_aggregation": "final_step_only_no_double_counting",
                "distractor_selection": "bm25_semantic"
            }
        }
        
        print(f"\nResults:")
        print(f"Precision@1: {precision_mean:.3f} [{precision_ci[0]:.3f}, {precision_ci[1]:.3f}]")
        print(f"Mean rank: {rank_mean:.2f} (median: {rank_median:.0f})")
        print(f"Gini coefficient: {gini_mean:.3f}")
        print(f"Success rate: {successful}/{attempted} ({100*successful/max(attempted, 1):.1f}%)")
        
        return results
