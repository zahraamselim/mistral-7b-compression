"""Document processing and text extraction."""

import re
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Process documents and extract clean text.
    Supports: PDF, TXT, Markdown
    """
    
    def __init__(self, config: dict):
        """
        Initialize document processor.
        
        Args:
            config: Document processing config from config.json
        """
        self.remove_headers = config.get('remove_headers', True)
        self.remove_citations = config.get('remove_citations', True)
        self.extract_sections = config.get('extract_sections', False)
        
        if PyPDF2 is None:
            logger.warning("PyPDF2 not installed. PDF processing will not work.")
    
    def process_file(self, filepath: str) -> List[Tuple[str, int]]:
        """
        Process a file and extract text with page numbers.
        
        Args:
            filepath: Path to file
            
        Returns:
            List of (text, page_number) tuples
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            return self.process_pdf(filepath)
        elif suffix in ['.txt', '.md', '.markdown']:
            return self.process_text(filepath)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def process_pdf(self, filepath: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page numbers.
        
        Args:
            filepath: Path to PDF file
            
        Returns:
            List of (text, page_number) tuples
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        pages = []
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        cleaned_text = self._clean_text(text)
                        if cleaned_text:
                            pages.append((cleaned_text, i + 1))
            
            logger.info(f"Extracted {len(pages)} pages from PDF")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
        
        return pages
    
    def process_text(self, filepath: str) -> List[Tuple[str, int]]:
        """
        Process plain text file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            List of (text, page_number) tuples (single page)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            cleaned_text = self._clean_text(text)
            
            if not cleaned_text:
                logger.warning(f"No text extracted from {filepath}")
                return []
            
            return [(cleaned_text, 1)]
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def process_string(self, text: str) -> str:
        """
        Process a string directly.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return self._clean_text(text)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (common patterns)
        if self.remove_headers:
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
            text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
            text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove citations like [1], (Smith et al., 2020)
        if self.remove_citations:
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Fix common OCR errors
        text = text.replace('ï¬', 'fi')
        text = text.replace('ï¬‚', 'fl')
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract major sections from documents.
        
        Args:
            text: Document text
            
        Returns:
            Dict mapping section names to text
        """
        sections = {}
        
        # Common section headers
        section_patterns = [
            r'^#{1,3}\s+(.+?)$',  # Markdown headers
            r'^([A-Z][A-Za-z\s]+):?\s*$',  # Title case headers
            r'^\d+\.?\s+([A-Z][A-Za-z\s]+)$',  # Numbered sections
        ]
        
        lines = text.split('\n')
        current_section = "Introduction"
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a header
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_text:
                        sections[current_section] = '\n'.join(current_text)
                    
                    current_section = match.group(1).strip()
                    current_text = []
                    is_header = True
                    break
            
            if not is_header:
                current_text.append(line)
        
        # Save last section
        if current_text:
            sections[current_section] = '\n'.join(current_text)
        
        return sections
