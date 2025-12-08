"""Core abstractions for LLM quantization evaluation."""

from core.model_interface import ModelInterface
from core.benchmark import Benchmark, BenchmarkResult
from core.result import Result

__all__ = [
    'ModelInterface',
    'Benchmark', 
    'BenchmarkResult',
    'Result',
]