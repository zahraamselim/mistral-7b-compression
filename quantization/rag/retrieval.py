"""Context retrieval from vector store."""

from typing import List, Dict, Optional
import numpy as np
import logging

from rag.indexing import VectorStore
from rag.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class ContextRetriever:
    """
    Retrieve relevant context for queries.
    Includes re-ranking and diversity mechanisms.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        config: dict
    ):
        """
        Initialize context retriever.
        
        Args:
            vector_store: VectorStore instance
            embedding_model: EmbeddingModel instance
            config: Retrieval config from config.json
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        self.top_k = config.get('top_k', 3)
        self.similarity_threshold = config.get('similarity_threshold', 0.0)
        self.rerank = config.get('rerank', False)
        self.diversity_penalty = config.get('diversity_penalty', 0.0)
        
        # Get distance metric from vector store
        self.distance_metric = self._get_distance_metric()
        logger.info(f"Using distance metric: {self.distance_metric}")
    
    def _get_distance_metric(self) -> str:
        """Get the distance metric used by the vector store."""
        try:
            if self.vector_store.collection:
                metadata = self.vector_store.collection.metadata
                return metadata.get('hnsw:space', 'cosine')
        except:
            pass
        return 'cosine'  # Default
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score [0, 1].
        
        ChromaDB distance ranges:
        - cosine: L2 distance in [0, 2] where 0=identical, 2=opposite
        - l2: Euclidean distance in [0, inf]
        - ip (inner product): Negative dot product in [-inf, inf]
        
        Args:
            distance: Distance from vector store
            
        Returns:
            Similarity score in [0, 1] where 1=most similar
        """
        if self.distance_metric == 'cosine':
            # For cosine space, ChromaDB returns L2 distance of normalized vectors
            # L2(a,b) = sqrt(2 - 2*cos(θ)) for normalized vectors
            # So: cos(θ) = 1 - (L2^2 / 2)
            # Clamp distance to [0, 2] and convert
            distance = max(0.0, min(2.0, distance))
            cosine_sim = 1.0 - (distance * distance / 2.0)
            return max(0.0, min(1.0, cosine_sim))
        
        elif self.distance_metric == 'l2':
            # For L2 distance, use exponential decay
            # similarity ≈ 1 / (1 + distance)
            return 1.0 / (1.0 + distance)
        
        elif self.distance_metric == 'ip':
            # Inner product: higher (less negative) is better
            # Normalize to [0, 1] assuming range [-2, 0] for normalized vectors
            return max(0.0, min(1.0, (distance + 2.0) / 2.0))
        
        else:
            logger.warning(f"Unknown distance metric: {self.distance_metric}, using default conversion")
            return max(0.0, 1.0 - (distance / 2.0))
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve (overrides config)
            filters: Metadata filters
            
        Returns:
            List of dicts with 'text', 'score', 'distance', 'metadata', 'chunk_id'
        """
        k = top_k or self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k * 2 if self.rerank else k,  # Get more if reranking
                where=filters
            )
            
            # Check if we got results
            if not results['ids'][0]:
                logger.warning("No results found for query")
                return []
            
            # Format results
            retrieved_chunks = []
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                similarity_score = self._distance_to_similarity(distance)
                
                chunk_data = {
                    'text': results['documents'][0][i],
                    'score': similarity_score,
                    'distance': distance,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'chunk_id': results['ids'][0][i]
                }
                
                # Apply similarity threshold
                if chunk_data['score'] >= self.similarity_threshold:
                    retrieved_chunks.append(chunk_data)
            
            if not retrieved_chunks:
                logger.warning(f"No chunks passed similarity threshold of {self.similarity_threshold}")
                return []
            
            # Re-rank if enabled
            if self.rerank and len(retrieved_chunks) > k:
                retrieved_chunks = self._rerank(query, retrieved_chunks, k)
            else:
                retrieved_chunks = retrieved_chunks[:k]
            
            # Apply diversity penalty if enabled
            if self.diversity_penalty > 0 and len(retrieved_chunks) > 1:
                retrieved_chunks = self._apply_diversity(retrieved_chunks)
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
    
    def get_context_string(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n"
    ) -> str:
        """
        Retrieve and format context as a single string.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            separator: Separator between chunks
            
        Returns:
            Formatted context string
        """
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return ""
        
        context_parts = [chunk['text'] for chunk in chunks]
        return separator.join(context_parts)
    
    def _rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """
        Re-ranking using BM25-style token overlap + semantic similarity.
        
        Args:
            query: Query string
            chunks: List of chunk dicts
            top_k: Number of top chunks to return
            
        Returns:
            Re-ranked chunks
        """
        query_tokens = set(query.lower().split())
        
        for chunk in chunks:
            chunk_tokens = set(chunk['text'].lower().split())
            
            # Token overlap (BM25-like)
            overlap = len(query_tokens & chunk_tokens)
            overlap_score = overlap / max(len(query_tokens), 1)
            
            # Combine with original semantic similarity
            # 70% semantic, 30% lexical
            chunk['rerank_score'] = chunk['score'] * 0.7 + overlap_score * 0.3
        
        # Sort by rerank score
        chunks.sort(key=lambda x: x.get('rerank_score', x['score']), reverse=True)
        return chunks[:top_k]
    
    def _apply_diversity(self, chunks: List[Dict]) -> List[Dict]:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity using embeddings.
        
        MMR = λ * Similarity(query, chunk) - (1-λ) * max(Similarity(chunk, selected))
        
        Args:
            chunks: List of chunk dicts
            
        Returns:
            Diversified chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        # Lambda parameter (similarity vs diversity tradeoff)
        lambda_param = 1.0 - self.diversity_penalty  # diversity_penalty in [0, 1]
        
        # Embed all chunks for similarity comparison
        chunk_texts = [c['text'] for c in chunks]
        chunk_embeddings = self.embedding_model.embed(chunk_texts)
        
        # Start with highest scoring chunk
        selected = [chunks[0]]
        selected_embeddings = [chunk_embeddings[0]]
        remaining_indices = list(range(1, len(chunks)))
        
        while len(selected) < len(chunks) and remaining_indices:
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining_indices:
                # Original relevance score
                relevance = chunks[idx]['score']
                
                # Calculate max similarity to already selected chunks
                max_sim = 0.0
                for sel_emb in selected_embeddings:
                    # Cosine similarity between embeddings
                    sim = np.dot(chunk_embeddings[idx], sel_emb) / (
                        np.linalg.norm(chunk_embeddings[idx]) * np.linalg.norm(sel_emb)
                    )
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(chunks[best_idx])
                selected_embeddings.append(chunk_embeddings[best_idx])
                remaining_indices.remove(best_idx)
            else:
                break
        
        return selected