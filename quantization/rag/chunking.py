"""Smart text chunking strategies."""

import re
from typing import List, Optional
from dataclasses import dataclass
import logging

# Improved NLTK handling
logger = logging.getLogger(__name__)

try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK punkt tokenizer downloaded successfully")
except ImportError:
    logger.error("NLTK not installed. Install with: pip install nltk")
    raise


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    tokens: Optional[int] = None


class TextChunker:
    """
    Smart text chunking with multiple strategies.
    
    Strategies:
    - semantic: Respects paragraph and sentence boundaries
    - sentence: Groups sentences up to chunk_size
    - fixed: Fixed-size chunks with overlap
    """
    
    def __init__(self, config: dict):
        """
        Initialize text chunker.
        
        Args:
            config: Chunking config from config.json
        """
        self.strategy = config.get('strategy', 'semantic')
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.min_chunk_size = config.get('min_chunk_size', 100)
        
        try:
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except Exception as e:
            logger.error(f"Failed to load NLTK sentence tokenizer: {e}")
            raise
        
        self._global_chunk_id = 0
        
        logger.info(f"Initialized TextChunker with strategy: {self.strategy}")
        logger.info(f"  Chunk size: {self.chunk_size}")
        logger.info(f"  Overlap: {self.chunk_overlap}")
    
    def chunk(self, text: str, page_num: Optional[int] = None) -> List[Chunk]:
        """
        Main chunking method - routes to appropriate strategy.
        
        Args:
            text: Text to chunk
            page_num: Optional page number
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []
        
        if self.strategy == "semantic":
            return self._semantic_chunking(text, page_num)
        elif self.strategy == "sentence":
            return self._sentence_chunking(text, page_num)
        elif self.strategy == "fixed":
            return self._fixed_size_chunking(text, page_num)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
    
    def reset_chunk_ids(self):
        """Reset chunk ID counter."""
        self._global_chunk_id = 0
    
    def _get_next_chunk_id(self) -> str:
        """Get next unique chunk ID."""
        chunk_id = f"chunk_{self._global_chunk_id}"
        self._global_chunk_id += 1
        return chunk_id
    
    def _semantic_chunking(self, text: str, page_num: Optional[int]) -> List[Chunk]:
        """
        Semantic chunking - respects paragraph and sentence boundaries.
        Best for structured documents.
        """
        chunks = []
        paragraphs = re.split(r'\n\n+', text)
        
        current_chunk = ""
        start_char = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para or len(para) < 20:
                continue
            
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                # Save current chunk if it's substantial
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        current_chunk.strip(),
                        start_char,
                        page_num
                    ))
                    
                    # Add overlap
                    overlap_text = self._get_overlap(current_chunk)
                    start_char = start_char + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + " "
                else:
                    start_char += len(current_chunk)
                    current_chunk = ""
            
            current_chunk += para + "\n\n"
        
        # Add final chunk
        if len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                start_char,
                page_num
            ))
        
        return chunks
    
    def _sentence_chunking(self, text: str, page_num: Optional[int]) -> List[Chunk]:
        """
        Sentence-based chunking - groups sentences up to chunk_size.
        Good for general text.
        """
        try:
            sentences = self.sent_tokenizer.tokenize(text)
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}. Falling back to simple split.")
            # Fallback to simple period-based split
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        start_char = 0
        
        for sent in sentences:
            if len(current_chunk) + len(sent) > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk.strip(),
                        start_char,
                        page_num
                    ))
                    start_char += len(current_chunk)
                    current_chunk = ""
            
            current_chunk += sent + " "
        
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk.strip(),
                start_char,
                page_num
            ))
        
        return chunks
    
    def _fixed_size_chunking(self, text: str, page_num: Optional[int]) -> List[Chunk]:
        """
        Fixed-size chunking with overlap.
        Simple but effective baseline.
        """
        chunks = []
        words = text.split()
        
        if not words:
            return []
        
        i = 0
        chunk_count = 0
        while i < len(words):
            # Get chunk_size words
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(self._create_chunk(
                chunk_text,
                chunk_count * (self.chunk_size - self.chunk_overlap),
                page_num
            ))
            
            chunk_count += 1
            # Move forward with overlap
            i += max(1, self.chunk_size - self.chunk_overlap)  # Ensure we always move forward
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        start_char: int,
        page_num: Optional[int]
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        return Chunk(
            text=text,
            chunk_id=self._get_next_chunk_id(),
            start_char=start_char,
            end_char=start_char + len(text),
            page_number=page_num,
            tokens=len(text.split())
        )
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of chunk."""
        words = text.split()
        if len(words) <= self.chunk_overlap:
            return text
        
        overlap_words = words[-self.chunk_overlap:]
        return ' '.join(overlap_words)
