"""
RAG Retrieval Evaluation - Optimized for Kaggle T4 (16GB VRAM)
Complete evaluation of RAG system: retrieval quality, answer quality, efficiency
"""

import time
import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter

logger = logging.getLogger(__name__)


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


class RAGEvaluator:
    """
    Optimized RAG system evaluator for T4 GPU.
    
    Measures:
    1. Retrieval Quality: How well does retrieval find relevant context?
       - Context sufficiency: Does context contain the answer?
       - Context precision: Relevance to query
       - Retrieval scores: Quality of similarity matching
    
    2. Answer Quality: How good are generated answers?
       - Exact Match: Strict correctness
       - F1 Score: Token-level overlap with ground truth
       - Faithfulness: Answers grounded in context
       - ROUGE (optional): n-gram overlap metrics
    
    3. Efficiency: System performance
       - Retrieval latency
       - Generation latency
       - Throughput
    
    4. RAG vs No-RAG: Does retrieval help?
       - Performance improvement with RAG
       - Answer quality comparison
    """
    
    def __init__(
        self,
        model_interface,
        rag_pipeline,
        config: dict = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model_interface: ModelInterface instance
            rag_pipeline: RAGPipeline instance
            config: Optional config dict
        """
        self.model_interface = model_interface
        self.rag_pipeline = rag_pipeline
        self.config = config or {}
        
        # Check for optional libraries
        self._check_optional_metrics()
        
        logger.info("RAG Evaluator initialized")
    
    def _check_optional_metrics(self):
        """Check for optional metric libraries."""
        # ROUGE
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
            self.rouge_available = True
        except ImportError:
            self.rouge_available = False
            logger.warning("rouge-score not available. Install: pip install rouge-score")
    
    def run(
        self,
        questions: List[str],
        ground_truth_answers: List[str],
        documents: Optional[List[str]] = None,
        compare_no_rag: bool = True,
        save_detailed: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run comprehensive RAG evaluation.
        
        Args:
            questions: Test questions
            ground_truth_answers: Ground truth answers
            documents: Documents to index (if not already indexed)
            compare_no_rag: Compare with no-RAG baseline
            save_detailed: Save per-question detailed results
            output_dir: Directory for detailed results
            
        Returns:
            Complete evaluation metrics
        """
        logger.info(f"Questions: {len(questions)}")
        
        # Index documents if provided
        if documents:
            logger.info(f"Indexing {len(documents)} documents...")
            self.rag_pipeline.index_documents(documents, show_progress=True)
        
        # Check indexing
        stats = self.rag_pipeline.get_stats()
        if stats['vector_store'].get('count', 0) == 0:
            raise ValueError("No documents indexed! Provide documents or index them first.")
        
        logger.info(f"Vector store: {stats['vector_store'].get('count', 0)} chunks indexed")
        
        # Storage for results
        detailed_results = [] if save_detailed else None
        
        rag_predictions = []
        no_rag_predictions = []
        contexts = []
        retrieved_chunks_list = []
        retrieval_times = []
        rag_gen_times = []
        no_rag_gen_times = []
        
        logger.info("Processing questions...")
        
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_answers)):
            try:
                # Retrieve context
                retrieval_start = time.perf_counter()
                retrieved_chunks = self.rag_pipeline.retrieve(question)
                retrieval_time = (time.perf_counter() - retrieval_start) * 1000
                retrieval_times.append(retrieval_time)
                
                retrieved_chunks_list.append(retrieved_chunks)
                context_str = '\n\n'.join([chunk['text'] for chunk in retrieved_chunks])
                contexts.append(context_str)
                
                # Generate RAG answer
                rag_gen_start = time.perf_counter()
                rag_answer = self.rag_pipeline.generate_answer(question, retrieved_chunks)
                rag_gen_time = (time.perf_counter() - rag_gen_start) * 1000
                rag_gen_times.append(rag_gen_time)
                rag_predictions.append(rag_answer)
                
                # Generate no-RAG answer if requested
                no_rag_answer = None
                no_rag_time = 0.0
                if compare_no_rag:
                    no_rag_start = time.perf_counter()
                    no_rag_answer = self.rag_pipeline.generator.generate_without_context(question)
                    no_rag_time = (time.perf_counter() - no_rag_start) * 1000
                    no_rag_gen_times.append(no_rag_time)
                    no_rag_predictions.append(no_rag_answer)
                
                # Save detailed results
                if save_detailed:
                    detailed_results.append({
                        'question_id': i + 1,
                        'question': question,
                        'ground_truth': ground_truth,
                        'rag_answer': rag_answer,
                        'no_rag_answer': no_rag_answer,
                        'num_chunks_retrieved': len(retrieved_chunks),
                        'avg_retrieval_score': float(np.mean([c.get('score', 0.0) for c in retrieved_chunks])),
                        'context_length_chars': len(context_str),
                        'retrieval_time_ms': retrieval_time,
                        'rag_generation_time_ms': rag_gen_time,
                        'no_rag_generation_time_ms': no_rag_time
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(questions)}")
            
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                rag_predictions.append("")
                if compare_no_rag:
                    no_rag_predictions.append("")
                contexts.append("")
                retrieved_chunks_list.append([])
                retrieval_times.append(0.0)
                rag_gen_times.append(0.0)
                if compare_no_rag:
                    no_rag_gen_times.append(0.0)
        
        logger.info("Computing metrics...")
        
        # Build results
        results = {}
        
        # 1. Retrieval quality
        logger.info("  Evaluating retrieval quality...")
        retrieval_metrics = self._evaluate_retrieval_quality(
            questions, contexts, ground_truth_answers, retrieved_chunks_list
        )
        results['retrieval_quality'] = retrieval_metrics
        
        # 2. Answer quality (RAG)
        logger.info("  Evaluating answer quality...")
        answer_metrics = self._evaluate_answer_quality(
            rag_predictions, ground_truth_answers, contexts
        )
        results['answer_quality'] = answer_metrics
        
        # 3. No-RAG comparison
        if compare_no_rag and no_rag_predictions:
            logger.info("  Evaluating no-RAG baseline...")
            no_rag_metrics = self._evaluate_answer_quality_simple(
                no_rag_predictions, ground_truth_answers
            )
            results['no_rag_baseline'] = no_rag_metrics
            
            # Calculate improvements
            results['rag_improvement'] = {
                'f1_gain': answer_metrics['f1'] - no_rag_metrics['f1'],
                'f1_gain_percent': ((answer_metrics['f1'] - no_rag_metrics['f1']) / max(no_rag_metrics['f1'], 0.01)) * 100,
                'em_gain': answer_metrics['exact_match'] - no_rag_metrics['exact_match']
            }
        
        # 4. Efficiency metrics
        logger.info("  Computing efficiency metrics...")
        efficiency_metrics = self._evaluate_efficiency(
            retrieval_times, rag_gen_times, no_rag_gen_times if compare_no_rag else None
        )
        results['efficiency'] = efficiency_metrics
        
        # 5. Metadata
        results['metadata'] = {
            'num_questions': len(questions),
            'num_chunks_indexed': stats['vector_store'].get('count', 0),
            'avg_chunks_per_query': float(np.mean([len(c) for c in retrieved_chunks_list])),
            'retrieval_config': self.rag_pipeline.retriever.top_k
        }
        
        # Convert to serializable
        results = convert_to_serializable(results)
        
        # Save detailed results
        if save_detailed and output_dir and detailed_results:
            self._save_detailed_results(detailed_results, results, output_dir)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _evaluate_retrieval_quality(
        self,
        questions: List[str],
        contexts: List[str],
        ground_truths: List[str],
        retrieved_chunks: List[List[Dict]]
    ) -> Dict:
        """Evaluate quality of retrieved contexts."""
        sufficiency_scores = []
        precision_scores = []
        coverage_scores = []
        retrieval_scores = []
        context_lengths = []
        
        for question, context, answer, chunks in zip(
            questions, contexts, ground_truths, retrieved_chunks
        ):
            if not context.strip():
                sufficiency_scores.append(0.0)
                precision_scores.append(0.0)
                coverage_scores.append(0.0)
                retrieval_scores.append(0.0)
                context_lengths.append(0)
                continue
            
            answer_lower = answer.lower()
            context_lower = context.lower()
            
            # Sufficiency: does context contain answer?
            if answer_lower in context_lower:
                sufficiency = 1.0
            else:
                # Token overlap
                answer_tokens = set(answer_lower.split())
                context_tokens = set(context_lower.split())
                overlap = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
                sufficiency = 1.0 if overlap >= 0.8 else overlap
            sufficiency_scores.append(sufficiency)
            
            # Precision: query-context relevance
            query_tokens = set(question.lower().split())
            context_tokens = set(context_lower.split())
            precision = len(query_tokens & context_tokens) / len(query_tokens) if query_tokens else 0.0
            precision_scores.append(precision)
            
            # Coverage: answer terms in context
            answer_tokens = set(answer_lower.split())
            coverage = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
            coverage_scores.append(coverage)
            
            # Average retrieval score
            if chunks:
                avg_score = np.mean([c.get('score', 0.0) for c in chunks])
                retrieval_scores.append(avg_score)
            else:
                retrieval_scores.append(0.0)
            
            # Context length
            context_lengths.append(len(context.split()))
        
        return {
            'context_sufficiency': float(np.mean(sufficiency_scores)),
            'context_precision': float(np.mean(precision_scores)),
            'answer_coverage': float(np.mean(coverage_scores)),
            'avg_retrieval_score': float(np.mean(retrieval_scores)),
            'avg_context_length_words': float(np.mean(context_lengths)),
            'retrieval_consistency': float(np.std(retrieval_scores))
        }
    
    def _evaluate_answer_quality(
        self,
        predictions: List[str],
        references: List[str],
        contexts: List[str]
    ) -> Dict:
        """Evaluate quality of generated answers."""
        metrics = {}
        
        # Exact Match
        exact_matches = [self._exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        metrics['exact_match'] = float(np.mean(exact_matches))
        
        # F1 Score
        f1_scores = [self._token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        metrics['f1'] = float(np.mean(f1_scores))
        metrics['f1_std'] = float(np.std(f1_scores))
        
        # Faithfulness: grounded in context
        faithfulness_scores = [
            self._token_overlap(pred, ctx) 
            for pred, ctx in zip(predictions, contexts)
        ]
        metrics['faithfulness'] = float(np.mean(faithfulness_scores))
        
        # Answer length
        metrics['avg_answer_length_words'] = float(np.mean([len(p.split()) for p in predictions]))
        
        # ROUGE scores (if available)
        if self.rouge_available:
            rouge_scores = self._compute_rouge(predictions, references)
            metrics.update(rouge_scores)
        
        return metrics
    
    def _evaluate_answer_quality_simple(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """Simple answer quality evaluation (for no-RAG baseline)."""
        exact_matches = [self._exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        f1_scores = [self._token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        
        return {
            'exact_match': float(np.mean(exact_matches)),
            'f1': float(np.mean(f1_scores)),
            'avg_answer_length_words': float(np.mean([len(p.split()) for p in predictions]))
        }
    
    def _evaluate_efficiency(
        self,
        retrieval_times: List[float],
        rag_gen_times: List[float],
        no_rag_gen_times: Optional[List[float]]
    ) -> Dict:
        """Evaluate efficiency metrics."""
        metrics = {}
        
        # Retrieval timing
        if retrieval_times:
            metrics['avg_retrieval_time_ms'] = float(np.mean(retrieval_times))
            metrics['retrieval_time_std_ms'] = float(np.std(retrieval_times))
        
        # RAG generation timing
        if rag_gen_times:
            metrics['avg_rag_generation_time_ms'] = float(np.mean(rag_gen_times))
            metrics['rag_generation_time_std_ms'] = float(np.std(rag_gen_times))
        
        # No-RAG timing and comparison
        if no_rag_gen_times:
            metrics['avg_no_rag_generation_time_ms'] = float(np.mean(no_rag_gen_times))
            
            # Overhead: RAG vs no-RAG
            rag_total = np.mean(retrieval_times) + np.mean(rag_gen_times)
            no_rag_total = np.mean(no_rag_gen_times)
            metrics['rag_overhead_ms'] = float(rag_total - no_rag_total)
            metrics['rag_overhead_percent'] = float((rag_total - no_rag_total) / no_rag_total * 100)
        
        return metrics
    
    def _exact_match(self, prediction: str, reference: str) -> float:
        """Exact match score (normalized)."""
        return float(prediction.lower().strip() == reference.lower().strip())
    
    def _token_f1(self, prediction: str, reference: str) -> float:
        """Token-level F1 score."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _token_overlap(self, text1: str, text2: str) -> float:
        """Token overlap ratio."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1:
            return 0.0
        
        return len(tokens1 & tokens2) / len(tokens1)
    
    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute ROUGE scores."""
        all_scores = [
            self.rouge_scorer.score(ref, pred) 
            for ref, pred in zip(references, predictions)
        ]
        
        return {
            'rouge1': float(np.mean([s['rouge1'].fmeasure for s in all_scores])),
            'rouge2': float(np.mean([s['rouge2'].fmeasure for s in all_scores])),
            'rougeL': float(np.mean([s['rougeL'].fmeasure for s in all_scores]))
        }
    
    def _save_detailed_results(
        self,
        detailed_results: List[Dict],
        summary: Dict,
        output_dir: str
    ):
        """Save detailed per-question results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving detailed results to {output_dir}...")
        
        # Save JSON
        json_path = output_path / 'rag_evaluation_detailed.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'per_question': detailed_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved: {json_path.name}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        # Retrieval quality
        logger.info("\nRETRIEVAL QUALITY:")
        retr = results['retrieval_quality']
        logger.info(f"  Context Sufficiency: {retr['context_sufficiency']:.3f}")
        logger.info(f"  Context Precision: {retr['context_precision']:.3f}")
        logger.info(f"  Answer Coverage: {retr['answer_coverage']:.3f}")
        logger.info(f"  Avg Retrieval Score: {retr['avg_retrieval_score']:.3f}")
        
        # Answer quality
        logger.info("\nANSWER QUALITY (RAG):")
        ans = results['answer_quality']
        logger.info(f"  Exact Match: {ans['exact_match']:.3f}")
        logger.info(f"  F1 Score: {ans['f1']:.3f} ± {ans.get('f1_std', 0):.3f}")
        logger.info(f"  Faithfulness: {ans['faithfulness']:.3f}")
        if 'rouge1' in ans:
            logger.info(f"  ROUGE-1: {ans['rouge1']:.3f}")
        
        # RAG improvement
        if 'rag_improvement' in results:
            logger.info("\nRAG vs NO-RAG:")
            imp = results['rag_improvement']
            baseline = results['no_rag_baseline']
            logger.info(f"  RAG F1: {ans['f1']:.3f} | Baseline F1: {baseline['f1']:.3f}")
            logger.info(f"  F1 Gain: {imp['f1_gain']:+.3f} ({imp['f1_gain_percent']:+.1f}%)")
            logger.info(f"  EM Gain: {imp['em_gain']:+.3f}")
        
        # Efficiency
        logger.info("\nEFFICIENCY:")
        eff = results['efficiency']
        logger.info(f"  Retrieval: {eff.get('avg_retrieval_time_ms', 0):.1f}ms")
        logger.info(f"  RAG Generation: {eff.get('avg_rag_generation_time_ms', 0):.1f}ms")
        if 'rag_overhead_ms' in eff:
            logger.info(f"  RAG Overhead: {eff['rag_overhead_ms']:.1f}ms ({eff['rag_overhead_percent']:.1f}%)")