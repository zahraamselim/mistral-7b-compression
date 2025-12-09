"""
Attention Drift Benchmark

Measures attention stability during generation in quantized models.
"""

import random
import numpy as np
import torch
import gc
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datasets import load_dataset
import re

from models.model_interface import ModelInterface, GenerationConfig


class AttentionDriftBenchmark:
    """
    Measures attention stability during text generation at document level.
    
    Metrics:
        - Mean drift: Average L1 distance between consecutive document attention distributions
        - Max drift: Maximum drift observed across generation
        - Drift by position: Drift at specific token positions
        - Drift from relevant doc: Specific drift from answer-containing document
    """
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 150,
        generation_positions: List[int] = None,
        num_documents: int = 5,
        layers_to_analyze: Optional[List[int]] = None,
        random_seed: int = 42
    ):
        """
        Initialize attention drift benchmark.
        
        Args:
            model: Model interface instance
            num_samples: Number of test samples
            generation_positions: Token positions to measure attention
            num_documents: Number of documents (1 relevant + rest distractors)
            layers_to_analyze: Which layers to analyze (None = middle 50%)
            random_seed: Random seed for reproducibility
        """
        self.model = model
        self.num_samples = num_samples
        self.generation_positions = sorted(generation_positions or [1, 5, 10, 20, 40])
        self.num_documents = num_documents
        self.layers_to_analyze = layers_to_analyze
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def _load_datasets(self) -> Tuple[any, List[Dict[str, str]]]:
        """Load Natural Questions dataset with Wikipedia corpus."""
        print("Loading Natural Questions...")
        queries = load_dataset("nq_open", split="validation")
        
        print("Loading Wikipedia corpus...")
        try:
            wiki_dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.en",
                split="train",
                streaming=True
            )
            
            passages = []
            for i, sample in enumerate(wiki_dataset):
                if i >= 800:
                    break
                
                text = sample.get('text', '')
                title = sample.get('title', '')
                if len(text) > 200:
                    passages.append({
                        'text': text,
                        'title': title
                    })
            
            if len(passages) < 50:
                raise RuntimeError("Insufficient Wikipedia passages loaded")
            
        except Exception as e:
            print(f"Failed to load wikimedia/wikipedia: {e}")
            print("Falling back to simple-wikipedia...")
            
            wiki_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:800]")
            passages = []
            for sample in wiki_dataset:
                text = sample.get('text', '')
                title = sample.get('title', '')
                if len(text) > 200:
                    passages.append({
                        'text': text,
                        'title': title
                    })
        
        print(f"Loaded {len(queries)} queries and {len(passages)} passages")
        return queries, passages
    
    def _extract_relevant_passage(
        self,
        answer: str,
        passages: List[Dict[str, str]]
    ) -> Optional[str]:
        """Extract passage that naturally contains the answer."""
        answer_lower = answer.lower()
        
        for passage in random.sample(passages, min(80, len(passages))):
            text = passage['text']
            if answer_lower in text.lower():
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
        
        queries, passages = self._load_datasets()
        
        drift_scores = []
        max_drifts = []
        drift_by_position = defaultdict(list)
        drift_from_relevant = []
        
        doc_marker_ids = [self.model.tokenizer.encode("\n\n", add_special_tokens=False)[0]]
        
        max_pos = max(self.generation_positions)
        config = GenerationConfig(
            max_new_tokens=max_pos,
            do_sample=False,
            temperature=1.0
        )
        
        successful = 0
        attempted = 0
        
        while successful < self.num_samples and attempted < len(queries):
            sample = queries[attempted]
            attempted += 1
            
            query = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not query or not answers:
                continue
            
            answer = answers[0]
            
            relevant_passage = self._extract_relevant_passage(answer, passages)
            if not relevant_passage:
                continue
            
            distractor_passages = []
            for passage in random.sample(passages, min(40, len(passages))):
                text = passage['text'][:300]
                if (answer.lower() not in text.lower() and 
                    text != relevant_passage and
                    len(distractor_passages) < self.num_documents - 1):
                    distractor_passages.append(text)
            
            if len(distractor_passages) < self.num_documents - 1:
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
                inputs = self.model.encode(prompt, max_length=2048)
                output = self.model.generate(prompt, config, return_attentions=True)
                
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
                    
                    mean_drift = float(np.mean(position_drifts))
                    max_drift = float(np.max(position_drifts))
                    
                    drift_scores.append(mean_drift)
                    max_drifts.append(max_drift)
                    
                    successful += 1
                
                del inputs, output, attention_sequence
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    continue
            except Exception:
                continue
            
            if successful % 30 == 0 and successful > 0:
                print(f"Progress: {successful}/{self.num_samples}")
        
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
        
        results = {
            "mean_drift": mean_drift,
            "mean_drift_ci_95_lower": mean_drift_ci[0],
            "mean_drift_ci_95_upper": mean_drift_ci[1],
            "max_drift_mean": max_drift_mean,
            "max_drift_std": max_drift_std,
            "drift_from_relevant_mean": relevant_drift_mean,
            "drift_from_relevant_std": relevant_drift_std,
            "drift_by_position": drift_by_pos_stats,
            "num_samples": len(drift_scores),
            "samples_attempted": attempted,
            "generation_positions": self.generation_positions,
            "methodology": {
                "drift_measurement": "document_level_not_token_level",
                "layers_analyzed": "middle_50_percent" if self.layers_to_analyze is None else self.layers_to_analyze,
                "attention_aggregation": "per_step_no_double_counting"
            }
        }
        
        print(f"Mean drift: {mean_drift:.4f} [{mean_drift_ci[0]:.4f}, {mean_drift_ci[1]:.4f}]")
        print(f"Max drift: {max_drift_mean:.4f}, Relevant drift: {relevant_drift_mean:.4f}")
        print(f"Success: {successful}/{attempted} ({100*successful/max(attempted, 1):.1f}%)")
        
        return results