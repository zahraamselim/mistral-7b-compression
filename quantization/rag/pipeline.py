"""End-to-end RAG pipeline orchestration."""

import time
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path

from rag.document_processing import DocumentProcessor
from rag.chunking import TextChunker, Chunk
from rag.embedding import EmbeddingModel
from rag.indexing import VectorStore
from rag.retrieval import ContextRetriever
from rag.generation import RAGGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline.
    Orchestrates all components.
    """
    
    def __init__(self, config: dict):
        """
        Initialize RAG pipeline.
        
        Args:
            config: RAG config from config.json
        """
        self.config = config
        
        # Initialize components (will be set up later)
        self.doc_processor = None
        self.chunker = None
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None
        self.generator = None
        
        logger.info("RAG Pipeline initialized")
    
    def setup(self, model_interface):
        """
        Setup all pipeline components.
        
        Args:
            model_interface: ModelInterface instance for generation
        """
        logger.info("Setting up RAG pipeline components...")
        
        # Document processor
        doc_config = self.config.get('document_processing', {})
        self.doc_processor = DocumentProcessor(doc_config)
        
        # Chunker
        chunk_config = self.config.get('chunking', {})
        self.chunker = TextChunker(chunk_config)
        
        # Embedding model
        embed_config = self.config.get('embedding', {})
        self.embedding_model = EmbeddingModel(embed_config)
        
        # Vector store
        store_config = self.config.get('vector_store', {})
        self.vector_store = VectorStore(store_config)
        
        # Retriever
        retrieval_config = self.config.get('retrieval', {})
        self.retriever = ContextRetriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            config=retrieval_config
        )
        
        # Generator
        generation_config = self.config.get('generation', {})
        self.generator = RAGGenerator(
            model_interface=model_interface,
            config=generation_config
        )
        
        logger.info("Pipeline setup complete!")
    
    def index_documents(
        self,
        documents: Union[str, List[str]],
        show_progress: bool = True
    ) -> float:
        """
        Index documents into vector store.
        
        Args:
            documents: Either:
                - Path to document file (PDF or TXT)
                - List of document strings
            show_progress: Show progress bars
            
        Returns:
            Processing time in seconds
        """
        start_time = time.time()
        
        # Handle file path vs string list
        if isinstance(documents, str) and Path(documents).exists():
            logger.info(f"Processing document file: {documents}")
            pages = self.doc_processor.process_file(documents)
            logger.info(f"Extracted {len(pages)} pages")
            
            # Chunk text
            all_chunks = []
            for text, page_num in pages:
                chunks = self.chunker.chunk(text, page_num=page_num)
                all_chunks.extend(chunks)
        
        elif isinstance(documents, list):
            logger.info(f"Processing {len(documents)} document strings")
            
            # Process and chunk each document
            all_chunks = []
            for i, doc in enumerate(documents):
                # Clean the text
                cleaned_text = self.doc_processor.process_string(doc)
                
                # Chunk it
                chunks = self.chunker.chunk(cleaned_text, page_num=i+1)
                all_chunks.extend(chunks)
        
        else:
            raise ValueError("documents must be a file path or list of strings")
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_chunks(
            all_chunks,
            show_progress=show_progress
        )
        
        # Create index
        self.vector_store.create_index(all_chunks, embeddings)
        
        processing_time = time.time() - start_time
        logger.info(f"Indexing complete in {processing_time:.2f}s")
        
        return processing_time
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            
        Returns:
            List of context dicts with 'text', 'score', 'metadata', 'chunk_id'
        """
        return self.retriever.retrieve(query, top_k=top_k)
    
    def validate_retrieval(self, query: str, expected_terms: List[str]) -> Dict:
        """Validate that retrieval finds expected terms."""
        chunks = self.retrieve(query, top_k=5)
        
        found_terms = []
        for term in expected_terms:
            for chunk in chunks:
                if term.lower() in chunk['text'].lower():
                    found_terms.append(term)
                    break
        
        return {
            'query': query,
            'expected': expected_terms,
            'found': found_terms,
            'recall': len(found_terms) / len(expected_terms),
            'chunks': chunks
        }

    def generate_answer(
        self,
        query: str,
        contexts: Optional[List[Dict]] = None,
        retrieve_if_none: bool = True
    ) -> str:
        """
        Generate answer for a query, optionally with provided contexts.
        
        Args:
            query: Query string
            contexts: Pre-retrieved contexts (if None, retrieves automatically)
            retrieve_if_none: If True and contexts is None, retrieve automatically
            
        Returns:
            Generated answer string
        """
        # If no contexts provided and we should retrieve
        if contexts is None and retrieve_if_none:
            contexts = self.retriever.retrieve(query)
        
        # Format context string
        if contexts and len(contexts) > 0:
            context_str = '\n\n'.join([
                ctx.get('text', ctx.get('content', ''))
                for ctx in contexts
            ])
        else:
            context_str = ""
        
        # Generate answer
        if context_str:
            return self.generator.generate(query, context_str)
        else:
            return self.generator.generate_without_context(query)
    
    def query(
        self,
        question: str,
        return_context: bool = False,
        return_chunks: bool = False
    ):
        """
        Query the RAG system.
        
        Args:
            question: User question
            return_context: If True, include context in return
            return_chunks: If True, include retrieved chunks in return
            
        Returns:
            Generated answer (or dict with answer, context, chunks)
        """
        # Retrieve context
        retrieved_chunks = self.retriever.retrieve(question)
        context = self.retriever.get_context_string(question)
        
        # Generate answer
        answer = self.generator.generate(question, context)
        
        if return_context or return_chunks:
            result = {'answer': answer}
            if return_context:
                result['context'] = context
            if return_chunks:
                result['chunks'] = retrieved_chunks
            return result
        
        return answer
    
    def evaluate(
        self,
        test_questions: List[Dict[str, str]],
        compare_no_rag: bool = True,
        show_progress: bool = True
    ) -> Dict:
        """
        Evaluate RAG system on test questions using batch generation.
        
        Args:
            test_questions: List of {'question': ..., 'answer': ...}
            compare_no_rag: Whether to compare with no-RAG baseline
            show_progress: Show progress bars
            
        Returns:
            Dict with predictions and optionally no-RAG predictions
        """
        logger.info(f"Evaluating on {len(test_questions)} questions")
        
        questions = [qa['question'] for qa in test_questions]
        references = [qa['answer'] for qa in test_questions]
        
        # Retrieve contexts for all questions
        logger.info("Retrieving contexts...")
        contexts = []
        contexts_list = []  # List of retrieved chunks for each question
        
        for q in questions:
            retrieved = self.retriever.retrieve(q)
            contexts_list.append(retrieved)
            context_str = '\n\n'.join([chunk['text'] for chunk in retrieved])
            contexts.append(context_str)
        
        # Generate RAG answers in batch
        logger.info("Generating RAG answers...")
        predictions = self.generator.generate_batch(
            queries=questions,
            contexts=contexts,
            show_progress=show_progress
        )
        
        # Generate no-RAG answers if needed
        predictions_no_rag = None
        if compare_no_rag:
            logger.info("Generating no-RAG baseline answers...")
            predictions_no_rag = self.generator.generate_batch_without_context(
                queries=questions,
                show_progress=show_progress
            )
        
        return {
            'questions': questions,
            'references': references,
            'predictions': predictions,
            'contexts': contexts,
            'retrieved_chunks': contexts_list,
            'predictions_no_rag': predictions_no_rag
        }
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            'vector_store': self.vector_store.get_stats() if self.vector_store else {},
            'embedding_dim': self.embedding_model.get_dimension() if self.embedding_model else None,
            'config': self.config
        }
        
        # Add embedding info
        if self.embedding_model:
            stats['embedding'] = {
                'model_name': self.embedding_model.model_name,
                'dimension': self.embedding_model.get_dimension(),
                'device': self.embedding_model.device,
                'batch_size': self.embedding_model.batch_size,
                'normalize': self.embedding_model.normalize
            }
        
        # Add retriever stats
        if self.retriever:
            stats['retrieval'] = {
                'top_k': self.retriever.top_k,
                'similarity_threshold': self.retriever.similarity_threshold,
                'rerank': self.retriever.rerank,
                'diversity_penalty': self.retriever.diversity_penalty,
                'distance_metric': self.retriever.distance_metric
            }
        
        return stats