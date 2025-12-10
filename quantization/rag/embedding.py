"""Text embedding generation."""

import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging

from rag.chunking import Chunk

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for embedding models.
    Supports sentence-transformers and custom models.
    """
    
    def __init__(self, config: dict):
        """
        Initialize embedding model.
        
        Args:
            config: Embedding config from config.json
        """
        self.model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.batch_size = config.get('batch_size', 32)
        self.normalize = config.get('normalize', True)
        self.device = self._get_device(config.get('device', 'cuda'))
        
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded on {self.device}")
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def _get_device(self, device_preference: str) -> str:
        """Determine best available device."""
        if device_preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device_preference == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Chunk], show_progress: bool = True) -> np.ndarray:
        """
        Embed list of Chunk objects.
        
        Args:
            chunks: List of Chunk objects
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed(texts, show_progress=show_progress)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
