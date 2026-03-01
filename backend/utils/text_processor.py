"""Optimized text processor with caching and better performance."""
from typing import List, Dict, Optional
import re
from functools import lru_cache
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    text: str
    start_word: int
    end_word: int
    word_count: int

@dataclass
class SearchResult:
    position: int
    context: str
    highlight_start: int
    highlight_end: int
    score: float = 1.0

class TextProcessor:
    def __init__(self, cache_size: int = 100):
        self._cache_size = cache_size
    
    @lru_cache(maxsize=100)
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[TextChunk]:
        """
        Chunk text into smaller pieces with optimized memory usage and caching.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum number of words per chunk
            overlap: Number of words to overlap between chunks
            
        Returns:
            List of TextChunk objects
        """
        try:
            # Split text into words efficiently
            words = text.split()
            chunks: List[TextChunk] = []
            
            # Process chunks with minimal memory allocation
            for i in range(0, len(words), chunk_size - overlap):
                end_idx = min(i + chunk_size, len(words))
                chunk_words = words[i:end_idx]
                
                chunks.append(TextChunk(
                    text=' '.join(chunk_words),
                    start_word=i,
                    end_word=end_idx,
                    word_count=len(chunk_words)
                ))
                
                if end_idx >= len(words):
                    break
            
            return chunks
        
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []

    @lru_cache(maxsize=100)
    def search_in_text(
        self, 
        text: str, 
        query: str, 
        context_length: int = 100,
        fuzzy_match: bool = True
    ) -> List[SearchResult]:
        """
        Improved text search with fuzzy matching and relevance scoring.
        
        Args:
            text: Text to search in
            query: Search query
            context_length: Number of characters for context
            fuzzy_match: Whether to use fuzzy matching
            
        Returns:
            List of SearchResult objects
        """
        try:
            if not query.strip() or not text:
                return []

            results: List[SearchResult] = []
            
            # Normalize text and query
            text_lower = text.lower()
            query_lower = query.lower()
            
            # Exact matches first
            exact_matches = self._find_exact_matches(
                text, text_lower, query, query_lower, context_length)
            results.extend(exact_matches)
            
            # Fuzzy matches if enabled and no exact matches found
            if fuzzy_match and not exact_matches:
                fuzzy_matches = self._find_fuzzy_matches(
                    text, text_lower, query, query_lower, context_length)
                results.extend(fuzzy_matches)
            
            # Sort by score and position
            results.sort(key=lambda x: (-x.score, x.position))
            
            return results[:10]  # Limit number of results
        
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []

    def _find_exact_matches(
        self, 
        text: str, 
        text_lower: str,
        query: str, 
        query_lower: str,
        context_length: int
    ) -> List[SearchResult]:
        """Find exact matches in text."""
        results = []
        start = 0
        
        while True:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break
            
            result = self._create_search_result(
                text, pos, len(query), context_length, score=1.0)
            results.append(result)
            
            start = pos + 1
        
        return results

    def _find_fuzzy_matches(
        self, 
        text: str, 
        text_lower: str,
        query: str, 
        query_lower: str,
        context_length: int
    ) -> List[SearchResult]:
        """Find fuzzy matches using word tokenization."""
        results = []
        query_words = query_lower.split()
        
        if not query_words:
            return results
        
        # Search for partial word matches
        words = text_lower.split()
        for i, word in enumerate(words):
            score = self._calculate_fuzzy_score(word, query_words)
            if score > 0.7:  # Threshold for fuzzy matches
                pos = text_lower.find(word)
                if pos != -1:
                    result = self._create_search_result(
                        text, pos, len(word), context_length, score)
                    results.append(result)
        
        return results

    def _create_search_result(
        self, 
        text: str, 
        pos: int, 
        match_length: int,
        context_length: int,
        score: float
    ) -> SearchResult:
        """Create a search result with context."""
        context_start = max(0, pos - context_length)
        context_end = min(len(text), pos + match_length + context_length)
        
        return SearchResult(
            position=pos,
            context=text[context_start:context_end],
            highlight_start=pos - context_start,
            highlight_end=pos - context_start + match_length,
            score=score
        )

    @staticmethod
    def _calculate_fuzzy_score(text: str, query_words: List[str]) -> float:
        """Calculate fuzzy matching score."""
        max_score = 0
        
        for query_word in query_words:
            if query_word in text:
                score = len(query_word) / len(text)
                max_score = max(max_score, score)
        
        return max_score

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()

    def get_text_stats(self, text: str) -> Dict[str, int]:
        """Get statistical information about text."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }

# Create a global instance
text_processor = TextProcessor()

# Expose main functions
def search_in_text(text: str, query: str, context_length: int = 100) -> List[Dict]:
    """Search for text matches with context."""
    results = text_processor.search_in_text(text, query, context_length)
    return [
        {
            'position': r.position,
            'context': r.context,
            'highlight_start': r.highlight_start,
            'highlight_end': r.highlight_end,
            'score': r.score
        }
        for r in results
    ]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Chunk text into smaller pieces."""
    chunks = text_processor.chunk_text(text, chunk_size, overlap)
    return [
        {
            'text': c.text,
            'start_word': c.start_word,
            'end_word': c.end_word,
            'word_count': c.word_count
        }
        for c in chunks
    ]

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching."""
    return text_processor.preprocess_text(text)

def get_text_stats(text: str) -> Dict[str, int]:
    """Get statistical information about text."""
    return text_processor.get_text_stats(text)