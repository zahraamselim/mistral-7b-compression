"""LLM answer generation with RAG - IMPROVED VERSION."""

import logging
import re
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generate answers using LLM with retrieved context.
    IMPROVED: Better balance between faithfulness and natural generation.
    """
    
    def __init__(self, model_interface, config: dict):
        """
        Initialize RAG generator.
        
        Args:
            model_interface: ModelInterface instance
            config: Generation config from config.json
        """
        self.model_interface = model_interface
        
        self.max_new_tokens = config.get('max_new_tokens', 128)
        self.temperature = config.get('temperature', 0.3)  # Raised from 0.1
        self.top_p = config.get('top_p', 0.9)
        self.do_sample = config.get('do_sample', True)  # Enable sampling
        self.repetition_penalty = config.get('repetition_penalty', 1.15)
        
        self.model_type = model_interface.model_type or "instruct"
        self.use_chat_template = config.get('use_chat_template', True)
        
        logger.info(f"Initialized RAG generator for {self.model_type} model")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Max tokens: {self.max_new_tokens}")
    
    def generate(
        self,
        query: str,
        context: str,
        return_prompt: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """
        Generate answer given query and context.
        
        Args:
            query: User question
            context: Retrieved context
            return_prompt: If True, return (answer, prompt) tuple
            
        Returns:
            Generated answer (or tuple with prompt)
        """
        # Truncate context if too long (prevent overwhelming the model)
        context = self._truncate_context(context, max_chars=2000)
        
        # Format prompt based on model type
        if self.model_type == "instruct":
            prompt = self._format_instruct_prompt(query, context)
        else:
            prompt = self._format_base_prompt(query, context)
        
        # Generate with balanced parameters
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
        answer = self._clean_answer(answer)
        
        # Light validation (less strict)
        if self._is_problematic(answer, context):
            logger.warning(f"Answer seems problematic: {answer[:80]}...")
            
            # Try once more with a clearer prompt
            logger.info("Retrying with simplified prompt...")
            simple_prompt = self._format_simple_prompt(query, context)
            answer = self.model_interface.generate(
                prompt=simple_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,  # Lower for retry
                do_sample=True,
                repetition_penalty=self.repetition_penalty
            )
            answer = self._clean_answer(answer)
        
        if return_prompt:
            return answer, prompt
        return answer
    
    def generate_batch(
        self,
        queries: List[str],
        contexts: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Generate answers for multiple queries in batch."""
        if len(queries) != len(contexts):
            raise ValueError(f"Queries ({len(queries)}) and contexts ({len(contexts)}) must have same length")
        
        answers = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(zip(queries, contexts), total=len(queries), desc="Generating answers")
            except ImportError:
                iterator = zip(queries, contexts)
                logger.warning("tqdm not available, progress bar disabled")
        else:
            iterator = zip(queries, contexts)
        
        for query, context in iterator:
            answer = self.generate(query, context)
            answers.append(answer)
        
        return answers
    
    def generate_without_context(self, query: str) -> str:
        """Generate answer without RAG context."""
        if self.model_type == "instruct":
            prompt = self._format_instruct_prompt_no_rag(query)
        else:
            prompt = self._format_base_prompt_no_rag(query)
        
        answer = self.model_interface.generate(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )
        
        return self._clean_answer(answer)
    
    def generate_batch_without_context(
        self,
        queries: List[str],
        show_progress: bool = True
    ) -> List[str]:
        """Generate answers for multiple queries without context."""
        answers = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(queries, desc="Generating no-RAG answers")
            except ImportError:
                iterator = queries
                logger.warning("tqdm not available, progress bar disabled")
        else:
            iterator = queries
        
        for query in iterator:
            answer = self.generate_without_context(query)
            answers.append(answer)
        
        return answers
    
    def _truncate_context(self, context: str, max_chars: int = 2000) -> str:
        """
        Truncate context to prevent overwhelming the model.
        Tries to keep complete sentences.
        """
        if len(context) <= max_chars:
            return context
        
        # Try to truncate at sentence boundary
        truncated = context[:max_chars]
        last_period = truncated.rfind('.')
        
        if last_period > max_chars * 0.7:  # If we can keep at least 70%
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _format_instruct_prompt(self, query: str, context: str) -> str:
        """
        IMPROVED: More natural prompt that encourages synthesis.
        """
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": f"""Use the following context to answer the question. Provide a clear, direct answer based on the information given.

Context:
{context}

Question: {query}

Answer:"""
            }]
            
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback format")
        
        # Fallback format (simpler)
        return f"""Context: {context}

Question: {query}

Based on the context above, provide a clear and concise answer:"""
    
    def _format_simple_prompt(self, query: str, context: str) -> str:
        """Simplified prompt for retry attempts."""
        return f"""Use this information to answer the question:

{context}

Question: {query}
Answer:"""
    
    def _format_base_prompt(self, query: str, context: str) -> str:
        """Format prompt for base models."""
        return f"""Context: {context}

Question: {query}

Answer:"""
    
    def _format_instruct_prompt_no_rag(self, query: str) -> str:
        """Format prompt without context for instruct models."""
        tokenizer = self.model_interface.get_tokenizer()
        
        if self.use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{
                "role": "user",
                "content": query
            }]
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")
        
        return f"Question: {query}\n\nAnswer:"
    
    def _format_base_prompt_no_rag(self, query: str) -> str:
        """Format prompt without context for base models."""
        return f"Question: {query}\n\nAnswer:"
    
    def _clean_answer(self, answer: str, max_sentences: int = 4) -> str:
        """
        Clean and normalize generated answer.
        """
        if not answer or answer.lower() in ['', 'none', 'n/a']:
            return "The information is not provided in the given context."
        
        # Remove excessive whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Remove common artifacts
        answer = re.sub(r'^(Answer:|A:)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        # Remove "Based on the context" prefix if it's redundant
        answer = re.sub(r'^Based on (the )?(context|information)( provided)?,?\s*', '', answer, flags=re.IGNORECASE)
        
        # Split into sentences
        sentences = re.split(r'([.!?])\s+', answer)
        
        # Reconstruct sentences properly
        reconstructed = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                reconstructed.append(sentences[i] + sentences[i + 1])
            else:
                reconstructed.append(sentences[i])
        
        # Filter out very short sentences
        reconstructed = [s for s in reconstructed if len(s.strip()) > 10]
        
        # Take only first N sentences
        if len(reconstructed) > max_sentences:
            answer = ' '.join(reconstructed[:max_sentences])
        else:
            answer = ' '.join(reconstructed)
        
        # Ensure it ends with punctuation
        if answer and not answer[-1] in '.!?':
            answer += '.'
        
        return answer
    
    def _is_problematic(self, answer: str, context: str) -> bool:
        """
        IMPROVED: Light validation to catch obvious problems only.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            True if answer has obvious problems
        """
        answer_lower = answer.lower()
        
        # Check for fallback/error responses
        if any(phrase in answer_lower for phrase in [
            "not provided", "not in the context", "cannot answer",
            "insufficient information", "does not specify"
        ]):
            return False  # These are fine
        
        # Check if answer is just repeating large chunks of context verbatim
        answer_clean = answer.lower().replace('.', '').replace(',', '').strip()
        context_clean = context.lower().replace('.', '').replace(',', '').strip()
        
        # Find longest common substring
        answer_words = answer_clean.split()
        context_words = context_clean.split()
        
        # Check for very long verbatim copies (10+ words in a row)
        for i in range(len(answer_words) - 10):
            answer_segment = ' '.join(answer_words[i:i+10])
            if answer_segment in context_clean:
                logger.warning("Answer contains long verbatim copy from context")
                return True
        
        # Check if answer is too short (< 15 words)
        if len(answer_words) < 15:
            logger.debug("Answer might be too short")
            return True
        
        # Check for repetition in answer itself
        if len(answer_words) >= 10:
            # Check if first half repeats in second half
            half = len(answer_words) // 2
            first_half = ' '.join(answer_words[:half])
            second_half = ' '.join(answer_words[half:])
            if first_half in second_half or second_half in first_half:
                logger.warning("Answer contains repetition")
                return True
        
        return False
