"""
RAG Benchmark Suite Runner

Orchestrates execution of all RAG-specific evaluation metrics.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from model_interface import ModelInterface
from rag.attention_preservation import AttentionPreservationBenchmark
from rag.context_degradation import ContextDegradationBenchmark
from rag.attention_drift import AttentionDriftBenchmark


class RAGBenchmarkSuite:
    """
    Complete RAG benchmark suite combining all novel metrics.
    
    Runs:
        1. Attention Preservation: Tests if model attends to relevant documents
        2. Context Degradation: Measures accuracy decline with context length
        3. Attention Drift: Measures attention stability during generation
    """
    
    def __init__(
        self,
        model: ModelInterface,
        attention_preservation_samples: int = 300,
        context_degradation_samples: int = 100,
        attention_drift_samples: int = 150
    ):
        """
        Initialize RAG benchmark suite.
        
        Args:
            model: Model interface instance
            attention_preservation_samples: Number of samples for attention preservation
            context_degradation_samples: Number of samples per context length
            attention_drift_samples: Number of samples for attention drift
        """
        self.model = model
        
        self.attention_preservation = AttentionPreservationBenchmark(
            model,
            num_samples=attention_preservation_samples,
            max_doc_tokens=128
        )
        
        self.context_degradation = ContextDegradationBenchmark(
            model,
            context_lengths=[512, 1024, 2048, 4096],
            samples_per_length=context_degradation_samples
        )
        
        self.attention_drift = AttentionDriftBenchmark(
            model,
            num_samples=attention_drift_samples,
            generation_positions=[1, 5, 10, 20, 40]
        )
    
    def _save_json(self, data: Dict, filepath: Path) -> None:
        """Save results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved to {filepath.name}")
    
    def _generate_summary(self, results: Dict[str, Dict]) -> Dict[str, any]:
        """Generate summary statistics across all benchmarks."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_info": self.model.get_model_info(),
            "benchmarks_run": list(results.keys()),
        }
        
        if "attention_preservation" in results:
            ap = results["attention_preservation"]
            summary["attention_preservation_summary"] = {
                "precision_at_1": ap.get("attention_precision_at_1"),
                "mean_rank": ap.get("attention_rank_mean"),
                "samples": ap.get("num_samples")
            }
        
        if "context_degradation" in results:
            cd = results["context_degradation"]
            summary["context_degradation_summary"] = {
                "degradation_slope_per_1k": cd.get("degradation_slope_per_1k_tokens"),
                "r_squared": cd.get("r_squared"),
                "significant": cd.get("interpretation") == "significant_degradation",
                "cliff_point": cd.get("cliff_point")
            }
        
        if "attention_drift" in results:
            ad = results["attention_drift"]
            summary["attention_drift_summary"] = {
                "mean_drift": ad.get("mean_drift"),
                "max_drift": ad.get("max_drift_mean"),
                "samples": ad.get("num_samples")
            }
        
        return summary
    
    def run_attention_preservation(
        self,
        output_dir: Optional[Path] = None
    ) -> Dict[str, any]:
        """Run attention preservation benchmark."""
        
        results = self.attention_preservation.run()
        
        if output_dir:
            self._save_json(results, output_dir / "attention_preservation.json")
        
        return results
    
    def run_context_degradation(
        self,
        output_dir: Optional[Path] = None
    ) -> Dict[str, any]:
        """Run context degradation benchmark."""
        
        results = self.context_degradation.run()
        
        if output_dir:
            self._save_json(results, output_dir / "context_degradation.json")
        
        return results
    
    def run_attention_drift(
        self,
        output_dir: Optional[Path] = None
    ) -> Dict[str, any]:
        """Run attention drift benchmark."""
        
        results = self.attention_drift.run()
        
        if output_dir:
            self._save_json(results, output_dir / "attention_drift.json")
        
        return results
    
    def run_all(
        self,
        output_dir: Path,
        save_intermediate: bool = True,
        skip_on_error: bool = False
    ) -> Dict[str, Dict]:
        """
        Run all RAG benchmarks.
        
        Args:
            output_dir: Directory to save results
            save_intermediate: Whether to save results after each benchmark
            skip_on_error: Whether to continue on benchmark failure
            
        Returns:
            Dictionary with all RAG metrics
        """
        print(f"\nRAG Benchmark Suite - Model: {self.model.model_path}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        benchmarks = [
            ("attention_preservation", self.run_attention_preservation),
            ("context_degradation", self.run_context_degradation),
            ("attention_drift", self.run_attention_drift),
        ]
        
        for i, (name, benchmark_func) in enumerate(benchmarks, 1):
            print(f"\n[{i}/{len(benchmarks)}] {name.replace('_', ' ').title()}")
            
            try:
                benchmark_output_dir = output_dir if save_intermediate else None
                benchmark_results = benchmark_func(output_dir=benchmark_output_dir)
                results[name] = benchmark_results
                
            except Exception as e:
                print(f"ERROR in {name}: {str(e)}")
                
                if skip_on_error:
                    print(f"Skipping and continuing...")
                    results[name] = {"error": str(e)}
                else:
                    raise
        
        summary = self._generate_summary(results)
        results["summary"] = summary
        
        self._save_json(results, output_dir / "rag_complete_results.json")
        self._save_json(summary, output_dir / "rag_summary.json")
        
        print("\nRAG Benchmark Complete")
        print(f"Results: {output_dir}")
        
        if "attention_preservation" in results:
            ap = results["attention_preservation"]
            print(f"\nAttention Preservation: Precision@1={ap.get('attention_precision_at_1', 0):.3f}, Rank={ap.get('attention_rank_mean', 0):.2f}")
        
        if "context_degradation" in results:
            cd = results["context_degradation"]
            print(f"Context Degradation: Slope={cd.get('degradation_slope_per_1k_tokens', 0):.4f}/1k, R²={cd.get('r_squared', 0):.4f}")
            if cd.get('cliff_point'):
                print(f"  Cliff point: {cd['cliff_point']} tokens")
        
        if "attention_drift" in results:
            ad = results["attention_drift"]
            print(f"Attention Drift: Mean={ad.get('mean_drift', 0):.4f}, Max={ad.get('max_drift_mean', 0):.4f}")
        
        return results
    
    def compare_models(
        self,
        baseline_results: Dict[str, Dict],
        output_dir: Path
    ) -> Dict[str, any]:
        """
        Compare current model results against baseline.
        
        Args:
            baseline_results: Results from baseline model
            output_dir: Directory to save comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        current_results = self.run_all(output_dir, save_intermediate=False)
        
        comparison = {
            "baseline_model": baseline_results.get("summary", {}).get("model_info"),
            "current_model": current_results.get("summary", {}).get("model_info"),
            "metrics_comparison": {}
        }
        
        if "attention_preservation" in baseline_results and "attention_preservation" in current_results:
            baseline_ap = baseline_results["attention_preservation"]
            current_ap = current_results["attention_preservation"]
            
            comparison["metrics_comparison"]["attention_preservation"] = {
                "precision_at_1_baseline": baseline_ap.get("attention_precision_at_1"),
                "precision_at_1_current": current_ap.get("attention_precision_at_1"),
                "precision_at_1_delta": (
                    current_ap.get("attention_precision_at_1", 0) - 
                    baseline_ap.get("attention_precision_at_1", 0)
                ),
                "rank_baseline": baseline_ap.get("attention_rank_mean"),
                "rank_current": current_ap.get("attention_rank_mean"),
                "rank_delta": (
                    current_ap.get("attention_rank_mean", 0) - 
                    baseline_ap.get("attention_rank_mean", 0)
                )
            }
        
        if "context_degradation" in baseline_results and "context_degradation" in current_results:
            baseline_cd = baseline_results["context_degradation"]
            current_cd = current_results["context_degradation"]
            
            comparison["metrics_comparison"]["context_degradation"] = {
                "slope_baseline": baseline_cd.get("degradation_slope_per_1k_tokens"),
                "slope_current": current_cd.get("degradation_slope_per_1k_tokens"),
                "slope_delta": (
                    current_cd.get("degradation_slope_per_1k_tokens", 0) - 
                    baseline_cd.get("degradation_slope_per_1k_tokens", 0)
                )
            }
        
        if "attention_drift" in baseline_results and "attention_drift" in current_results:
            baseline_ad = baseline_results["attention_drift"]
            current_ad = current_results["attention_drift"]
            
            comparison["metrics_comparison"]["attention_drift"] = {
                "drift_baseline": baseline_ad.get("mean_drift"),
                "drift_current": current_ad.get("mean_drift"),
                "drift_delta": (
                    current_ad.get("mean_drift", 0) - 
                    baseline_ad.get("mean_drift", 0)
                )
            }
        
        self._save_json(comparison, output_dir / "model_comparison.json")
        
        return comparison