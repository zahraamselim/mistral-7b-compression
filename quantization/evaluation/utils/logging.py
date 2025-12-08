"""Logging configuration utilities."""

import logging
import sys
import warnings
from typing import Optional, List


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    suppress_libraries: Optional[List[str]] = None,
    suppress_warnings: bool = True
):
    """
    Setup unified logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
        suppress_libraries: Library names to suppress
        suppress_warnings: Whether to suppress Python warnings
    """
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    
    handler = logging.StreamHandler(sys.stderr)
    
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    
    root.addHandler(handler)
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    root.setLevel(level_map.get(level.upper(), logging.INFO))
    
    default_suppress = [
        'chromadb',
        'sentence_transformers',
        'transformers',
        'urllib3',
        'httpx',
        'httpcore',
        'filelock',
        'huggingface_hub',
        'torch.distributed'
    ]
    
    if suppress_libraries is not None:
        libraries_to_suppress = suppress_libraries
    else:
        libraries_to_suppress = default_suppress
    
    for lib in libraries_to_suppress:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    if suppress_warnings:
        warnings.filterwarnings('ignore')


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with optional level override.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional level override
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    return logger