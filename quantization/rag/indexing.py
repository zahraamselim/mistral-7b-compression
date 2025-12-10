"""Vector store creation and management."""

import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any
import numpy as np
import logging

from rag.chunking import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store using ChromaDB.
    Handles storage, indexing, and similarity search.
    """
    
    def __init__(self, config: dict):
        """
        Initialize vector store.
        
        Args:
            config: Vector store config from config.json
        """
        self.collection_name = config.get('collection_name', 'rag_documents')
        self.persist_directory = config.get('persist_directory', None)
        
        # Initialize client
        try:
            if self.persist_directory:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                logger.info(f"Using persistent storage: {self.persist_directory}")
            else:
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
                logger.info("Using in-memory storage")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
        
        # Get or create collection
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or load collection."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            logger.info(f"Collection size: {self.collection.count()}")
        except Exception:
            # Collection doesn't exist yet
            logger.info(f"Collection '{self.collection_name}' will be created on first add")
    
    def create_index(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        metadata_fields: Optional[List[str]] = None
    ):
        """
        Create vector index from chunks and embeddings.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Embedding vectors (numpy array)
            metadata_fields: Which chunk fields to include in metadata
        """
        if len(chunks) == 0:
            logger.warning("No chunks provided for indexing")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})")
        
        # Create collection if it doesn't exist
        if self.collection is None:
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection: {e}")
                raise
        
        # Prepare data
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        
        # Build metadata
        if metadata_fields is None:
            metadata_fields = ['page_number', 'section', 'tokens']
        
        metadatas = []
        for chunk in chunks:
            meta = {}
            for field in metadata_fields:
                value = getattr(chunk, field, None)
                if value is not None:
                    # ChromaDB requires metadata values to be strings, ints, or floats
                    if isinstance(value, (str, int, float)):
                        meta[field] = value
                    else:
                        meta[field] = str(value)
            metadatas.append(meta)
        
        # Add to collection
        try:
            logger.info(f"Adding {len(chunks)} chunks to index...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Index created successfully! Total documents: {self.collection.count()}")
        except Exception as e:
            logger.error(f"Failed to add documents to collection: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector (numpy array)
            top_k: Number of results to return
            where: Metadata filters (e.g., {"page_number": 5})
            where_document: Document content filters
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances'
        """
        if self.collection is None:
            raise ValueError("No collection available. Create index first.")
        
        if self.collection.count() == 0:
            logger.warning("Collection is empty. No results to return.")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Ensure top_k doesn't exceed collection size
        collection_size = self.collection.count()
        top_k = min(top_k, collection_size)
        
        # Convert embedding to list properly
        if isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_emb_list = query_embedding.tolist()
            else:
                query_emb_list = query_embedding.reshape(-1).tolist()
        elif isinstance(query_embedding, list):
            query_emb_list = query_embedding
        else:
            query_emb_list = list(query_embedding)
        
        # Ensure it's a list of lists (batch format) for ChromaDB
        if not isinstance(query_emb_list[0], list):
            query_emb_list = [query_emb_list]
        
        try:
            results = self.collection.query(
                query_embeddings=query_emb_list,
                n_results=top_k,
                where=where,
                where_document=where_document
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def delete_collection(self):
        """Delete the collection."""
        if self.collection:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = None
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to delete collection: {e}")
                raise
    
    def reset_collection(self):
        """Reset the collection (delete and recreate)."""
        self.delete_collection()
        self._initialize_collection()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None:
            return {"status": "empty", "count": 0}
        
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "error": str(e)}
