"""Base benchmark class."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import logging

from core.result import BenchmarkResult

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BenchmarkResult')


class Benchmark(ABC, Generic[T]):
    """
    Abstract base class for all benchmarks.
    
    All benchmark implementations should inherit from this.
    """
    
    def __init__(
        self,
        model_interface,
        config: dict,
        verbose: bool = False
    ):
        """
        Initialize benchmark.
        
        Args:
            model_interface: ModelInterface instance
            config: Benchmark configuration
            verbose: Enable verbose logging
        """
        self.model_interface = model_interface
        self.model = model_interface.get_model()
        self.tokenizer = model_interface.get_tokenizer()
        self.config = config
        self.verbose = verbose
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    @abstractmethod
    def run(self, **kwargs) -> T:
        """
        Run benchmark and return results.
        
        Returns:
            BenchmarkResult subclass instance
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate benchmark configuration.
        
        Returns:
            True if valid
        """
        if self.config is None:
            logger.warning("No configuration provided")
            return False
        return True
    
    def validate_result(self, result: T) -> bool:
        """
        Validate benchmark result.
        
        Args:
            result: BenchmarkResult instance
            
        Returns:
            True if valid
        """
        if result is None:
            logger.error("Result is None")
            return False
        
        return result.validate()