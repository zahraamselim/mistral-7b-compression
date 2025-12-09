"""
RAG-Specific Evaluation Metrics

This module implements novel evaluation metrics specifically designed for 
assessing quantized language models in Retrieval-Augmented Generation scenarios.

Metrics:
    - Attention Preservation: Measures attention accuracy to relevant documents
      * Multi-layer attention aggregation (middle 50% of layers)
      * Proper document-level attention across all generated tokens
      * BM25-based semantic distractor selection
      
    - Context Degradation: Tracks performance decline with context length
      * Natural answer embedding (no artificial markers)
      * Token-level pre-tokenized context construction
      * Tests true long-context reasoning capabilities
      
    - Attention Drift: Measures attention stability during generation
      * Document-level drift measurement (not token-level)
      * Multi-layer attention tracking across generation
      * Drift from relevant document specifically measured

All metrics use real datasets and extract actual attention weights from model internals.

Key improvements over standard approaches:
    - Attention aggregation across all tokens (not just last token)
    - Middle-layer analysis (where retrieval attention occurs)
    - Natural text construction (no artificial markers)
    - Statistical rigor (bootstrap CIs, power analysis)
"""

from .attention_preservation import AttentionPreservationBenchmark
from .context_degradation import ContextDegradationBenchmark
from .attention_drift import AttentionDriftBenchmark
from .runner import RAGBenchmarkSuite

__all__ = [
    "AttentionPreservationBenchmark",
    "ContextDegradationBenchmark",
    "AttentionDriftBenchmark",
    "RAGBenchmarkSuite",
]
