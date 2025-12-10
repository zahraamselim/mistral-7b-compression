"""
Attention Evaluation Suite - HIGHLY OPTIMIZED

Unified attention analysis for RAG models combining:
1. Attention Preservation: Focus on relevant documents
2. Attention Drift: Stability during generation

Key optimizations:
- Single evaluation loop for both benchmarks
- Shared dataset loading and indexing
- Unified document processing
- Single generation pass captures both metrics
- Memory-efficient attention extraction
"""

import random
import numpy as np
import torch
import gc
import time
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import re
from collections import defaultdict

from model_interface import ModelInterface, GenerationConfig


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


def compute_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction matches any ground truth."""
    pred = prediction.lower().strip()
    return any(gt.lower().strip() in pred or pred in gt.lower().strip() for gt in ground_truths)


class DatasetCache:
    """Shared dataset cache for attention evaluation."""
    
    def __init__(self):
        self.nq_dataset = None
        self.wiki_passages = None
        self.answer_index = None
        self.bm25_index = None
    
    def load(self, max_docs: int = 400):
        """Load and index datasets once."""
        if self.nq_dataset is not None:
            return
        
        print("Loading datasets...")
        self.nq_dataset = load_dataset("nq_open", split="validation")
        print(f"  {len(self.nq_dataset)} questions")
        
        try:
            wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
            self.wiki_passages = []
            for i, s in enumerate(wiki):
                if i >= max_docs:
                    break
                if len(s.get('text', '')) > 200:
                    self.wiki_passages.append({
                        'text': s['text'],
                        'text_lower': s['text'].lower()
                    })
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1} wiki docs...")
        except Exception:
            wiki = load_dataset("wikipedia", "20220301.simple", split=f"train[:{max_docs}]")
            self.wiki_passages = [
                {'text': s['text'], 'text_lower': s['text'].lower()}
                for s in wiki if len(s.get('text', '')) > 200
            ]
        
        print(f"  {len(self.wiki_passages)} passages")
        
        self.answer_index = defaultdict(list)
        for idx, p in enumerate(self.wiki_passages):
            for i in range(len(p['text_lower'].split()[:80]) - 2):
                phrase = ' '.join(p['text_lower'].split()[i:i+3])
                self.answer_index[phrase].append(idx)
        print(f"  {len(self.answer_index)} phrases indexed")
        
        corpus = [p['text'][:600].lower().split()[:80] for p in self.wiki_passages]
        self.bm25_index = BM25Okapi(corpus)
        print("  BM25 indexed\n")


class AttentionEvaluator:
    """
    Unified attention evaluator capturing both preservation and drift.
    
    Single evaluation loop measures:
    - Preservation: Does attention focus on relevant documents?
    - Drift: Does attention remain stable during generation?
    """
    
    @staticmethod
    def create_fast(model: ModelInterface):
        """Fast: 40 samples, 3 generation steps"""
        return AttentionEvaluator(
            model, num_samples=40, generation_positions=[1, 5, 20], num_documents=5
        )
    
    @staticmethod
    def create_standard(model: ModelInterface):
        """Standard: 80 samples, 3 generation steps"""
        return AttentionEvaluator(
            model, num_samples=80, generation_positions=[1, 5, 20], num_documents=5
        )
    
    def __init__(
        self,
        model: ModelInterface,
        num_samples: int = 80,
        generation_positions: List[int] = None,
        num_documents: int = 5
    ):
        self.model = model
        self.cache = DatasetCache()
        self.num_samples = num_samples
        self.generation_positions = sorted(generation_positions or [1, 5, 20])
        self.num_documents = num_documents
        self.num_layers = None
        self.doc_marker_ids = None
    
    def _extract_context(self, answer: str) -> Optional[str]:
        """Find passage containing answer."""
        answer_lower = answer.lower()
        words = answer_lower.split()
        
        if len(words) >= 3:
            phrase = ' '.join(words[:3])
            if phrase in self.cache.answer_index:
                candidates = self.cache.answer_index[phrase]
                if candidates:
                    passage = self.cache.wiki_passages[random.choice(candidates[:8])]
                    context = self._extract_sentences(passage['text'], answer_lower)
                    if context:
                        return context
        
        for p in random.sample(self.cache.wiki_passages, min(80, len(self.cache.wiki_passages))):
            if answer_lower in p['text_lower']:
                context = self._extract_sentences(p['text'], answer_lower)
                if context:
                    return context
        
        return None
    
    def _extract_sentences(self, text: str, answer: str) -> Optional[str]:
        """Extract 2-3 sentences around answer."""
        sentences = re.split(r'[.!?]+\s+', text)
        for i, sent in enumerate(sentences):
            if answer in sent.lower():
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context = '. '.join(sentences[start:end])
                if len(context) > 50:
                    return context
        return None
    
    def _select_distractors(self, query: str, answer: str, relevant: str) -> List[str]:
        """BM25-based distractor selection."""
        scores = self.cache.bm25_index.get_scores(query.lower().split()[:20])
        top_indices = np.argsort(scores)[::-1]
        
        answer_lower = answer.lower()
        selected = []
        
        for idx in top_indices:
            if len(selected) >= self.num_documents - 1:
                break
            text = self.cache.wiki_passages[idx]['text'][:600]
            if answer_lower not in text.lower() and text != relevant:
                sentences = re.split(r'[.!?]+', text)
                if len(sentences) > 2:
                    selected.append('. '.join(sentences[:3]))
        
        while len(selected) < self.num_documents - 1:
            p = random.choice(self.cache.wiki_passages)
            text = p['text'][:400]
            if text not in selected and answer_lower not in text.lower():
                selected.append(text)
        
        return selected[:self.num_documents - 1]
    
    def _get_doc_spans(self, tokens: torch.Tensor) -> List[Tuple[int, int]]:
        """Get token span for each document."""
        tokens_list = tokens[0].cpu().tolist()
        spans = []
        pos = 0
        
        for _ in range(self.num_documents):
            markers = [j for j in range(pos, len(tokens_list)) if tokens_list[j] in self.doc_marker_ids]
            if len(markers) >= 2:
                spans.append((markers[0], markers[1]))
                pos = markers[1]
            elif len(markers) == 1:
                spans.append((markers[0], len(tokens_list)))
                break
            else:
                break
        
        while len(spans) < self.num_documents:
            spans.append((0, 0))
        
        return spans
    
    def _aggregate_attention(
        self,
        attentions: Tuple[torch.Tensor],
        input_ids: torch.Tensor,
        num_generated: int = 0
    ) -> Optional[np.ndarray]:
        """Aggregate token attention to document level."""
        if not attentions:
            return None
        
        spans = self._get_doc_spans(input_ids)
        start_layer = self.num_layers // 4
        end_layer = 3 * self.num_layers // 4
        
        doc_attn = np.zeros(self.num_documents)
        total = 0.0
        
        for layer_idx in range(start_layer, end_layer):
            if layer_idx >= len(attentions):
                continue
            
            try:
                attn = attentions[layer_idx].mean(dim=1).squeeze()
                
                if attn.dim() == 1:
                    token_attn = attn.cpu().numpy()
                    for doc_idx, (s, e) in enumerate(spans):
                        if s < e <= len(token_attn):
                            val = token_attn[s:e].sum()
                            doc_attn[doc_idx] += val
                            total += val
                else:
                    gen_start = max(0, attn.shape[0] - num_generated) if num_generated > 0 else 0
                    for t in range(gen_start, attn.shape[0]):
                        token_attn = attn[t, :].cpu().numpy()
                        for doc_idx, (s, e) in enumerate(spans):
                            if s < e <= len(token_attn):
                                val = token_attn[s:e].sum()
                                doc_attn[doc_idx] += val
                                total += val
                
                del attn
            except Exception:
                continue
        
        return doc_attn / total if total > 0 else None
    
    def _calculate_drift(self, attn1: np.ndarray, attn2: np.ndarray) -> float:
        """L1 distance between attention distributions."""
        if attn1.sum() > 0:
            attn1 = attn1 / attn1.sum()
        if attn2.sum() > 0:
            attn2 = attn2 / attn2.sum()
        return float(np.abs(attn2 - attn1).sum())
    
    def run(self) -> Dict[str, any]:
        """Run unified attention evaluation."""
        print(f"Unified Attention Evaluation: {self.num_samples} samples\n")
        
        self.cache.load()
        self.num_layers = self.model.get_model_info()['num_layers']
        self.doc_marker_ids = [self.model.tokenizer.encode("\n\n", add_special_tokens=False)[0]]
        
        # Preservation metrics
        precision_at_1 = []
        attention_ranks = []
        attention_gini = []
        
        # Drift metrics
        drift_scores = []
        max_drifts = []
        drift_from_relevant = []
        
        # Quality metrics
        exact_matches = []
        f1_scores = []
        
        # Correlations
        attn_when_correct = []
        attn_when_wrong = []
        drift_when_correct = []
        drift_when_wrong = []
        relevant_drift_when_correct = []
        relevant_drift_when_wrong = []
        
        max_pos = max(self.generation_positions)
        config = GenerationConfig(max_new_tokens=max_pos, do_sample=False)
        
        successful = 0
        attempted = 0
        start_time = time.time()
        
        while successful < self.num_samples and attempted < len(self.cache.nq_dataset):
            iter_start = time.time()
            sample = self.cache.nq_dataset[attempted]
            attempted += 1
            
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            
            if not question or not answers:
                print(f"[{successful}/{self.num_samples}] Attempt {attempted}: Skip (no Q/A)")
                continue
            
            relevant_doc = self._extract_context(answers[0])
            if not relevant_doc:
                print(f"[{successful}/{self.num_samples}] Attempt {attempted}: Skip (no context)")
                continue
            
            distractors = self._select_distractors(question, answers[0], relevant_doc)
            if len(distractors) < self.num_documents - 1:
                print(f"[{successful}/{self.num_samples}] Attempt {attempted}: Skip (no distractors)")
                continue
            
            all_docs = [relevant_doc[:500]] + [d[:500] for d in distractors]
            random.shuffle(all_docs)
            answer_idx = all_docs.index(relevant_doc[:500])
            
            prompt = f"Question: {question}\n\n" + "\n\n".join([
                f"Document {i+1}: {d}" for i, d in enumerate(all_docs)
            ]) + "\n\nAnswer:"
            
            try:
                inputs = self.model.encode(prompt, max_length=2048)
                output = self.model.generate(prompt, config, return_attentions=True)
                
                # Quality metrics
                em = compute_exact_match(output.generated_text, answers)
                f1 = max([compute_f1(output.generated_text, ans) for ans in answers])
                exact_matches.append(1.0 if em else 0.0)
                f1_scores.append(f1)
                is_correct = em or f1 > 0.3
                
                if not output.attentions or len(output.attentions) < min(self.generation_positions):
                    print(f"[{successful}/{self.num_samples}] Attempt {attempted}: Skip (no attentions)")
                    del inputs, output
                    continue
                
                # Extract attention at multiple generation steps
                attention_sequence = []
                for pos in self.generation_positions:
                    if pos > len(output.attentions):
                        break
                    doc_attn = self._aggregate_attention(
                        output.attentions[pos - 1], inputs['input_ids'], pos
                    )
                    if doc_attn is not None:
                        attention_sequence.append(doc_attn)
                
                if not attention_sequence:
                    print(f"[{successful}/{self.num_samples}] Attempt {attempted}: Skip (no valid attention)")
                    del inputs, output
                    continue
                
                # Preservation metrics (using final attention)
                final_attn = attention_sequence[-1]
                top_doc = np.argmax(final_attn)
                precision_at_1.append(1.0 if top_doc == answer_idx else 0.0)
                
                rank = int(np.where(np.argsort(final_attn)[::-1] == answer_idx)[0][0] + 1)
                attention_ranks.append(rank)
                
                sorted_attn = np.sort(final_attn)
                gini = (2 * np.sum((np.arange(len(sorted_attn)) + 1) * sorted_attn)) / (
                    len(sorted_attn) * sorted_attn.sum()
                ) - (len(sorted_attn) + 1) / len(sorted_attn)
                attention_gini.append(float(gini))
                
                relevant_attn = final_attn[answer_idx]
                if is_correct:
                    attn_when_correct.append(relevant_attn)
                else:
                    attn_when_wrong.append(relevant_attn)
                
                # Drift metrics (if multiple steps)
                if len(attention_sequence) > 1:
                    drifts = []
                    for j in range(1, len(attention_sequence)):
                        drift = self._calculate_drift(attention_sequence[j-1], attention_sequence[j])
                        drifts.append(drift)
                        
                        relevant_drift = abs(
                            attention_sequence[j][answer_idx] - attention_sequence[j-1][answer_idx]
                        )
                        drift_from_relevant.append(float(relevant_drift))
                        
                        if is_correct:
                            relevant_drift_when_correct.append(float(relevant_drift))
                        else:
                            relevant_drift_when_wrong.append(float(relevant_drift))
                    
                    mean_drift = float(np.mean(drifts))
                    drift_scores.append(mean_drift)
                    max_drifts.append(float(np.max(drifts)))
                    
                    if is_correct:
                        drift_when_correct.append(mean_drift)
                    else:
                        drift_when_wrong.append(mean_drift)
                
                successful += 1
                
                iter_time = time.time() - iter_start
                avg_time = (time.time() - start_time) / successful
                eta = (self.num_samples - successful) * avg_time
                
                print(f"[{successful}/{self.num_samples}] Attempt {attempted}: "
                      f"Prec@1={np.mean(precision_at_1):.3f} "
                      f"Drift={np.mean(drift_scores):.4f if drift_scores else 0:.4f} "
                      f"F1={np.mean(f1_scores):.3f} "
                      f"Time={iter_time:.1f}s ETA={eta/60:.1f}min")
                
                del inputs, output, attention_sequence
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"[{successful}/{self.num_samples}] Attempt {attempted}: OOM")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"[{successful}/{self.num_samples}] Attempt {attempted}: Error ({type(e).__name__})")
                continue
        
        if not precision_at_1:
            raise ValueError("No successful evaluations")
        
        # Bootstrap CIs
        def bootstrap_ci(data):
            means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(1000)]
            return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        
        precision_ci = bootstrap_ci(precision_at_1)
        drift_ci = bootstrap_ci(drift_scores) if drift_scores else (0.0, 0.0)
        
        # Correlations
        preservation_corr = {}
        if len(precision_at_1) == len(f1_scores):
            pr, pp = pearsonr(precision_at_1, f1_scores)
            sr, sp = spearmanr(attention_ranks, f1_scores)
            preservation_corr = {
                "precision_vs_f1_pearson_r": float(pr),
                "precision_vs_f1_pearson_p": float(pp),
                "rank_vs_f1_spearman_r": float(sr),
                "rank_vs_f1_spearman_p": float(sp)
            }
        
        drift_corr = {}
        if drift_scores and len(drift_scores) == len(f1_scores):
            dr, dp = pearsonr(drift_scores, f1_scores)
            drift_corr = {
                "drift_vs_f1_pearson_r": float(dr),
                "drift_vs_f1_pearson_p": float(dp)
            }
        
        results = {
            "preservation": {
                "precision_at_1": float(np.mean(precision_at_1)),
                "precision_ci_95": precision_ci,
                "rank_mean": float(np.mean(attention_ranks)),
                "rank_median": float(np.median(attention_ranks)),
                "gini_mean": float(np.mean(attention_gini)),
                "attention_when_correct": float(np.mean(attn_when_correct)) if attn_when_correct else 0.0,
                "attention_when_wrong": float(np.mean(attn_when_wrong)) if attn_when_wrong else 0.0,
                "correlation": preservation_corr
            },
            "drift": {
                "mean_drift": float(np.mean(drift_scores)) if drift_scores else 0.0,
                "drift_ci_95": drift_ci,
                "max_drift_mean": float(np.mean(max_drifts)) if max_drifts else 0.0,
                "drift_from_relevant": float(np.mean(drift_from_relevant)) if drift_from_relevant else 0.0,
                "drift_when_correct": float(np.mean(drift_when_correct)) if drift_when_correct else 0.0,
                "drift_when_wrong": float(np.mean(drift_when_wrong)) if drift_when_wrong else 0.0,
                "relevant_drift_when_correct": float(np.mean(relevant_drift_when_correct)) if relevant_drift_when_correct else 0.0,
                "relevant_drift_when_wrong": float(np.mean(relevant_drift_when_wrong)) if relevant_drift_when_wrong else 0.0,
                "correlation": drift_corr
            },
            "quality": {
                "exact_match": float(np.mean(exact_matches)),
                "f1_mean": float(np.mean(f1_scores))
            },
            "metadata": {
                "num_samples": successful,
                "samples_attempted": attempted,
                "generation_positions": self.generation_positions
            }
        }
        
        total_time = (time.time() - start_time) / 60
        print(f"\nCompleted in {total_time:.1f} min")
        print(f"Preservation - Precision@1: {results['preservation']['precision_at_1']:.3f}, Rank: {results['preservation']['rank_mean']:.2f}")
        print(f"Drift - Mean: {results['drift']['mean_drift']:.4f}, Max: {results['drift']['max_drift_mean']:.4f}")
        print(f"Quality - EM: {results['quality']['exact_match']:.3f}, F1: {results['quality']['f1_mean']:.3f}\n")
        
        return results