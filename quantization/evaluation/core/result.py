"""Base result classes with statistical support."""

from abc import ABC
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.debug("scipy not available for advanced statistics")


@dataclass
class Result:
    """Base class for result data."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {filepath}")


@dataclass
class BenchmarkResult(Result, ABC):
    """Base class for benchmark results with validation."""
    
    def validate(self) -> bool:
        """
        Validate result data.
        
        Returns:
            True if valid
        """
        result_dict = self.to_dict()
        
        if all(v is None for v in result_dict.values()):
            logger.error("Result contains only None values")
            return False
        
        for key, value in result_dict.items():
            if isinstance(value, float):
                if np.isnan(value):
                    logger.warning(f"Metric '{key}' is NaN")
                    return False
                if np.isinf(value):
                    logger.warning(f"Metric '{key}' is infinite")
                    return False
        
        return True
    
    def __str__(self) -> str:
        """Pretty print results."""
        lines = [f"\n{'='*60}", f"{self.__class__.__name__}", '='*60]
        
        def format_value(v, indent=0):
            if isinstance(v, float):
                return f"{v:.4f}"
            elif isinstance(v, dict):
                result = []
                for k, nested_v in v.items():
                    prefix = "  " * (indent + 1)
                    if isinstance(nested_v, dict):
                        result.append(f"{prefix}{k}:")
                        result.append(format_value(nested_v, indent + 1))
                    elif isinstance(nested_v, float):
                        result.append(f"{prefix}{k:.<36} {nested_v:.4f}")
                    else:
                        result.append(f"{prefix}{k:.<36} {nested_v}")
                return '\n'.join(result)
            else:
                return str(v)
        
        for key, value in self.to_dict().items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                lines.append(format_value(value, 0))
            elif isinstance(value, float):
                lines.append(f"{key:.<40} {value:.4f}")
            else:
                lines.append(f"{key:.<40} {value}")
        
        lines.append('='*60)
        return '\n'.join(lines)
    
    @staticmethod
    def aggregate_from_runs(
        runs: List['BenchmarkResult'],
        metrics: Optional[List[str]] = None,
        confidence: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results from multiple runs.
        
        Args:
            runs: List of BenchmarkResult instances
            metrics: Specific metrics to aggregate
            confidence: Confidence level for intervals
            
        Returns:
            Dict with statistics for each metric
        """
        if not runs:
            raise ValueError("No runs provided")
        
        all_metrics = {}
        for run in runs:
            for key, value in run.to_dict().items():
                if isinstance(value, (int, float)):
                    if metrics is None or key in metrics:
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(float(value))
        
        aggregated = {}
        for metric, values in all_metrics.items():
            values_array = np.array(values)
            n = len(values)
            mean = np.mean(values_array)
            std = np.std(values_array, ddof=1)
            std_err = std / np.sqrt(n)
            
            stats_dict = {
                'mean': float(mean),
                'std': float(std),
                'std_err': float(std_err),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'n_runs': n
            }
            
            if SCIPY_AVAILABLE and n > 1:
                t_val = scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
                ci_margin = t_val * std_err
                stats_dict['ci_lower'] = float(mean - ci_margin)
                stats_dict['ci_upper'] = float(mean + ci_margin)
                stats_dict['confidence_level'] = confidence
            
            aggregated[metric] = stats_dict
        
        return aggregated