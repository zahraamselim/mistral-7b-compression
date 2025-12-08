"""Statistical comparison tool for evaluation results."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Install: pip install scipy")


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two models."""
    metric: str
    baseline_value: float
    comparison_value: float
    absolute_diff: float
    relative_change_pct: float
    improved: bool
    statistically_significant: bool = False
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


class ResultsComparator:
    """Statistical comparison between evaluation results."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize comparator.
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level
        self.results = []
        self.result_names = []
    
    def load_result(self, filepath: str, name: Optional[str] = None):
        """Load a single result file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {filepath}")
        
        with open(path, 'r') as f:
            result = json.load(f)
        
        result_name = name or path.stem
        self.results.append(result)
        self.result_names.append(result_name)
        
        logger.info(f"Loaded result: {result_name}")
    
    def load_results(self, filepaths: List[str], names: Optional[List[str]] = None):
        """Load multiple result files."""
        if names and len(names) != len(filepaths):
            raise ValueError("Length of names must match filepaths")
        
        for i, filepath in enumerate(filepaths):
            name = names[i] if names else None
            self.load_result(filepath, name)
    
    def _find_metric_value(self, result: Dict, metric: str) -> Optional[float]:
        """Find a metric value in nested dictionary."""
        if metric in result and isinstance(result[metric], (int, float)):
            return float(result[metric])
        
        for key, value in result.items():
            if isinstance(value, dict) and metric in value:
                nested_val = value[metric]
                if isinstance(nested_val, (int, float)):
                    return float(nested_val)
        
        return None
    
    def compare_two(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: Optional[List[str]] = None
    ) -> List[ComparisonResult]:
        """
        Compare two results statistically.
        
        Args:
            baseline_idx: Index of baseline result
            comparison_idx: Index of comparison result
            metrics: List of metrics to compare
            
        Returns:
            List of ComparisonResult objects
        """
        if baseline_idx >= len(self.results) or comparison_idx >= len(self.results):
            raise IndexError("Invalid result index")
        
        baseline = self.results[baseline_idx]
        comparison = self.results[comparison_idx]
        
        if metrics is None:
            metrics = self._detect_numeric_metrics(baseline, comparison)
        
        comparisons = []
        
        for metric in metrics:
            baseline_val = self._find_metric_value(baseline, metric)
            comparison_val = self._find_metric_value(comparison, metric)
            
            if baseline_val is None or comparison_val is None:
                logger.warning(f"Metric '{metric}' not found in both results")
                continue
            
            abs_diff = comparison_val - baseline_val
            rel_change = (abs_diff / baseline_val * 100) if baseline_val != 0 else float('inf')
            
            is_higher_better = self._is_higher_better(metric)
            improved = (abs_diff > 0) if is_higher_better else (abs_diff < 0)
            
            comparison_result = ComparisonResult(
                metric=metric,
                baseline_value=baseline_val,
                comparison_value=comparison_val,
                absolute_diff=abs_diff,
                relative_change_pct=rel_change,
                improved=improved
            )
            
            comparisons.append(comparison_result)
        
        return comparisons
    
    def statistical_significance_test(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: List[str],
        alpha: float = 0.05,
        test_type: str = 'ttest'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Test if differences are statistically significant.
        
        Args:
            baseline_idx: Index of baseline result
            comparison_idx: Index of comparison result
            metrics: List of metrics to test
            alpha: Significance level
            test_type: 'ttest', 'welch', or 'ztest'
            
        Returns:
            Dict with significance test results for each metric
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy required for significance testing. Install: pip install scipy")
            return {}
        
        if baseline_idx >= len(self.results) or comparison_idx >= len(self.results):
            raise IndexError("Invalid result index")
        
        baseline = self.results[baseline_idx]
        comparison = self.results[comparison_idx]
        
        test_results = {}
        
        for metric in metrics:
            baseline_val = self._find_metric_value(baseline, metric)
            comparison_val = self._find_metric_value(comparison, metric)
            
            if baseline_val is None or comparison_val is None:
                logger.warning(f"Metric '{metric}' not found in both results")
                continue
            
            baseline_std = self._find_metric_value(baseline, f"{metric}_std")
            comparison_std = self._find_metric_value(comparison, f"{metric}_std")
            
            if baseline_std is None or comparison_std is None:
                logger.warning(f"No std deviation found for '{metric}'")
                continue
            
            baseline_n = self._find_metric_value(baseline, f"{metric}_n_runs")
            if baseline_n is None:
                baseline_n = 10
            baseline_n = int(baseline_n)
            
            comparison_n = self._find_metric_value(comparison, f"{metric}_n_runs")
            if comparison_n is None:
                comparison_n = 10
            comparison_n = int(comparison_n)
            
            if test_type in ['ttest', 'welch']:
                se_diff = np.sqrt((baseline_std**2 / baseline_n) + (comparison_std**2 / comparison_n))
                t_stat = (comparison_val - baseline_val) / se_diff if se_diff > 0 else 0.0
                
                df = ((baseline_std**2 / baseline_n + comparison_std**2 / comparison_n)**2 /
                      ((baseline_std**2 / baseline_n)**2 / (baseline_n - 1) +
                       (comparison_std**2 / comparison_n)**2 / (comparison_n - 1)))
                
                p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
                statistic = t_stat
                test_name = "Welch's t-test"
                
            elif test_type == 'ztest':
                se_diff = np.sqrt((baseline_std**2 / baseline_n) + (comparison_std**2 / comparison_n))
                z_stat = (comparison_val - baseline_val) / se_diff if se_diff > 0 else 0.0
                p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
                statistic = z_stat
                test_name = "Z-test"
            else:
                logger.warning(f"Unknown test type: {test_type}")
                continue
            
            pooled_std = np.sqrt((baseline_std**2 + comparison_std**2) / 2)
            cohens_d = (comparison_val - baseline_val) / pooled_std if pooled_std > 0 else 0.0
            
            test_results[metric] = {
                'test': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'alpha': alpha,
                'cohens_d': float(cohens_d),
                'effect_interpretation': self._interpret_cohens_d(abs(cohens_d)),
                'baseline_mean': float(baseline_val),
                'baseline_std': float(baseline_std),
                'baseline_n': int(baseline_n),
                'comparison_mean': float(comparison_val),
                'comparison_std': float(comparison_std),
                'comparison_n': int(comparison_n)
            }
        
        return test_results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def print_significance_test(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: List[str],
        alpha: float = 0.05
    ):
        """Print formatted significance test results."""
        results = self.statistical_significance_test(
            baseline_idx, comparison_idx, metrics, alpha
        )
        
        if not results:
            print("\nNo significance tests performed. Ensure results contain std/variance information.")
            print("Run benchmarks with multiple iterations to enable statistical testing.")
            return
        
        baseline_name = self.result_names[baseline_idx]
        comparison_name = self.result_names[comparison_idx]
        
        print("")
        print("Statistical Significance Test")
        print(f"{comparison_name} vs {baseline_name} (baseline)")
        print("")
        
        for metric, test_result in results.items():
            print(f"{metric}:")
            print(f"  Test: {test_result['test']}")
            print(f"  Baseline: {test_result['baseline_mean']:.4f} +/- {test_result['baseline_std']:.4f} (n={test_result['baseline_n']})")
            print(f"  Comparison: {test_result['comparison_mean']:.4f} +/- {test_result['comparison_std']:.4f} (n={test_result['comparison_n']})")
            print(f"  Statistic: {test_result['statistic']:.4f}")
            print(f"  p-value: {test_result['p_value']:.4f}")
            
            if test_result['significant']:
                print(f"  SIGNIFICANT (p < {test_result['alpha']})")
            else:
                print(f"  Not significant (p >= {test_result['alpha']})")
            
            print(f"  Effect size (Cohen's d): {test_result['cohens_d']:.4f} ({test_result['effect_interpretation']})")
            print("")
    
    def _detect_numeric_metrics(
        self,
        result1: Dict,
        result2: Dict
    ) -> List[str]:
        """Detect common numeric metrics between two results."""
        metrics = set()
        
        def extract_numeric_keys(d: Dict, prefix: str = ""):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (int, float)):
                    metrics.add(full_key)
                elif isinstance(value, dict):
                    extract_numeric_keys(value, full_key)
        
        extract_numeric_keys(result1)
        
        result2_metrics = set()
        extract_numeric_keys(result2)
        
        return sorted(list(metrics & result2_metrics))
    
    def _is_higher_better(self, metric: str) -> bool:
        """Determine if higher values are better for a metric."""
        higher_better = ['accuracy', 'precision', 'recall', 'f1', 'throughput',
                        'mfu', 'score', 'tokens_per_sec', 'speedup',
                        'improvement', 'exact_match']
        
        lower_better = ['latency', 'perplexity', 'loss', 'memory', 'energy',
                       'ms_per_token', 'time', 'error', 'std']
        
        metric_lower = metric.lower()
        
        if any(hb in metric_lower for hb in higher_better):
            return True
        
        if any(lb in metric_lower for lb in lower_better):
            return False
        
        return True
    
    def print_comparison(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: Optional[List[str]] = None,
        show_all: bool = False
    ):
        """Print formatted comparison results."""
        comparisons = self.compare_two(baseline_idx, comparison_idx, metrics)
        
        baseline_name = self.result_names[baseline_idx]
        comparison_name = self.result_names[comparison_idx]
        
        print("")
        print(f"Comparison: {comparison_name} vs {baseline_name} (baseline)")
        print("")
        
        improvements = [c for c in comparisons if c.improved]
        regressions = [c for c in comparisons if not c.improved and c.absolute_diff != 0]
        unchanged = [c for c in comparisons if c.absolute_diff == 0]
        
        print(f"Summary: {len(improvements)} improvements, {len(regressions)} regressions, {len(unchanged)} unchanged")
        
        if improvements:
            print("")
            print("Improvements:")
            for comp in sorted(improvements, key=lambda x: abs(x.relative_change_pct), reverse=True):
                self._print_comparison_line(comp)
        
        if regressions:
            print("")
            print("Regressions:")
            for comp in sorted(regressions, key=lambda x: abs(x.relative_change_pct), reverse=True):
                self._print_comparison_line(comp)
        
        if show_all and unchanged:
            print("")
            print("Unchanged:")
            for comp in unchanged:
                self._print_comparison_line(comp)
    
    def _print_comparison_line(self, comp: ComparisonResult):
        """Print a single comparison line."""
        symbol = "↑" if comp.improved else "↓"
        
        if abs(comp.relative_change_pct) < 0.01:
            change_str = f"({comp.absolute_diff:+.4f})"
        elif abs(comp.relative_change_pct) > 1000:
            change_str = f"({comp.absolute_diff:+.2f})"
        else:
            change_str = f"({comp.relative_change_pct:+.2f}%)"
        
        print(f"  {symbol} {comp.metric:<35} "
              f"{comp.baseline_value:>10.4f} -> {comp.comparison_value:>10.4f} "
              f"{change_str:>12}")
    
    def get_summary_statistics(
        self,
        baseline_idx: int,
        comparison_idx: int,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for comparison."""
        comparisons = self.compare_two(baseline_idx, comparison_idx, metrics)
        
        improvements = [c for c in comparisons if c.improved]
        regressions = [c for c in comparisons if not c.improved and c.absolute_diff != 0]
        
        if improvements:
            avg_improvement = np.mean([abs(c.relative_change_pct) for c in improvements])
            max_improvement = max(improvements, key=lambda x: abs(x.relative_change_pct))
        else:
            avg_improvement = 0.0
            max_improvement = None
        
        if regressions:
            avg_regression = np.mean([abs(c.relative_change_pct) for c in regressions])
            max_regression = max(regressions, key=lambda x: abs(x.relative_change_pct))
        else:
            avg_regression = 0.0
            max_regression = None
        
        return {
            'num_metrics': len(comparisons),
            'num_improvements': len(improvements),
            'num_regressions': len(regressions),
            'avg_improvement_pct': avg_improvement,
            'avg_regression_pct': avg_regression,
            'max_improvement': {
                'metric': max_improvement.metric,
                'change_pct': max_improvement.relative_change_pct
            } if max_improvement else None,
            'max_regression': {
                'metric': max_regression.metric,
                'change_pct': max_regression.relative_change_pct
            } if max_regression else None
        }
    
    def find_best_model(
        self,
        metric: str,
        higher_is_better: Optional[bool] = None
    ) -> Tuple[int, float, str]:
        """Find the best performing model for a metric."""
        if higher_is_better is None:
            higher_is_better = self._is_higher_better(metric)
        
        best_idx = None
        best_value = float('-inf') if higher_is_better else float('inf')
        
        for i, result in enumerate(self.results):
            value = self._find_metric_value(result, metric)
            if value is None:
                continue
            
            if higher_is_better:
                if value > best_value:
                    best_value = value
                    best_idx = i
            else:
                if value < best_value:
                    best_value = value
                    best_idx = i
        
        if best_idx is None:
            raise ValueError(f"Metric '{metric}' not found in any results")
        
        return best_idx, best_value, self.result_names[best_idx]
    
    def create_leaderboard(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """Create a weighted leaderboard across multiple metrics."""
        if weights is None:
            weights = [1.0] * len(metrics)
        
        if len(weights) != len(metrics):
            raise ValueError("Number of weights must match number of metrics")
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        scores = []
        
        for i, (result, name) in enumerate(zip(self.results, self.result_names)):
            score = 0.0
            valid_metrics = 0
            
            for metric, weight in zip(metrics, weights):
                value = self._find_metric_value(result, metric)
                if value is None:
                    continue
                
                all_values = [self._find_metric_value(r, metric) for r in self.results]
                all_values = [v for v in all_values if v is not None]
                
                if not all_values:
                    continue
                
                min_val = min(all_values)
                max_val = max(all_values)
                
                if max_val == min_val:
                    normalized = 1.0
                else:
                    normalized = (value - min_val) / (max_val - min_val)
                    
                    if not self._is_higher_better(metric):
                        normalized = 1.0 - normalized
                
                score += normalized * weight
                valid_metrics += 1
            
            if valid_metrics > 0:
                scores.append((name, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def print_leaderboard(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None
    ):
        """Print formatted leaderboard."""
        leaderboard = self.create_leaderboard(metrics, weights)
        
        print("")
        print("Leaderboard")
        print("")
        print(f"Metrics: {', '.join(metrics)}")
        if weights:
            print(f"Weights: {', '.join(f'{w:.2f}' for w in weights)}")
        print("")
        
        for rank, (name, score) in enumerate(leaderboard, 1):
            print(f"{rank}. {name:<40} Score: {score:.4f}")


def main():
    """CLI interface for comparator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare evaluation results')
    parser.add_argument('files', nargs='+', help='JSON result files')
    parser.add_argument('--names', nargs='+', help='Custom names for results')
    parser.add_argument('--baseline', type=int, default=0,
                       help='Index of baseline result (default: 0)')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to compare')
    parser.add_argument('--show-all', action='store_true',
                       help='Show all metrics including unchanged')
    parser.add_argument('--leaderboard', nargs='+',
                       help='Create leaderboard for these metrics')
    parser.add_argument('--weights', nargs='+', type=float,
                       help='Weights for leaderboard metrics')
    parser.add_argument('--best', help='Find best model for this metric')
    parser.add_argument('--significance', nargs='+',
                       help='Test statistical significance for these metrics')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level (default: 0.05)')
    
    args = parser.parse_args()
    
    comparator = ResultsComparator()
    comparator.load_results(args.files, args.names)
    
    if args.leaderboard:
        comparator.print_leaderboard(args.leaderboard, args.weights)
    elif args.best:
        idx, value, name = comparator.find_best_model(args.best)
        print(f"\nBest model for '{args.best}': {name}")
        print(f"Value: {value:.4f}")
    elif args.significance:
        if len(comparator.results) < 2:
            print("Need at least 2 results for significance testing")
        else:
            comparator.print_significance_test(
                args.baseline, 1, args.significance, args.alpha
            )
    else:
        for i in range(len(comparator.results)):
            if i != args.baseline:
                comparator.print_comparison(
                    args.baseline, i, args.metrics, args.show_all
                )
                print()


if __name__ == "__main__":
    main()