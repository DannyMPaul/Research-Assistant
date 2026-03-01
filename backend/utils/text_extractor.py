"""Optimized text extractor with better memory management and error handling."""
from typing import Optional, Dict, Any
import fitz
from docx import Document
import os
import logging
from pathlib import Path
import mmap
import chardet
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = ['extract_text', 'is_supported_file']

class TextExtractor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._supported_extensions = {'.pdf', '.docx', '.txt'}
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported."""
        return Path(file_path).suffix.lower() in self._supported_extensions
    
    async def extract_text(self, file_path: str) -> str:
        """
        Extract text from a file with optimized memory usage and error handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        try:
            file_path = str(Path(file_path).resolve())
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension not in self._supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Use appropriate extraction method based on file type
            if extension == '.pdf':
                return await self._extract_text_from_pdf(file_path)
            elif extension == '.docx':
                return await self._extract_text_from_docx(file_path)
            elif extension == '.txt':
                return await self._extract_text_from_txt(file_path)
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    async def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with optimized memory usage."""
        try:
            def extract():
                text_parts = []
                with fitz.open(file_path) as doc:
                    # Process pages in chunks to manage memory
                    chunk_size = 10
                    for i in range(0, len(doc), chunk_size):
                        chunk = doc[i:min(i + chunk_size, len(doc))]
                        for page in chunk:
                            # Extract text with optimal settings
                            text = page.get_text(
                                sort=True,  # Maintain reading order
                                flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE
                            )
                            text_parts.append(text)
                            
                return '\n'.join(text_parts)
            
            # Run in thread pool to avoid blocking
            return await self._run_in_executor(extract)
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    async def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX with optimized processing."""
        try:
            def extract():
                text_parts = []
                doc = Document(file_path)
                
                # Process paragraphs
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_parts.append(paragraph.text)
                
                # Process tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            text_parts.append(row_text)
                
                return '\n'.join(text_parts)
            
            return await self._run_in_executor(extract)
        
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    async def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT with encoding detection and memory mapping."""
        try:
            def extract():
                # Detect file encoding
                encoding = self._detect_encoding(file_path)
                
                # Use memory mapping for large files
                with open(file_path, 'r', encoding=encoding) as f:
                    if os.path.getsize(file_path) > 1024 * 1024:  # 1MB
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            return mm.read().decode(encoding)
                    else:
                        return f.read()
            
            return await self._run_in_executor(extract)
        
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding with caching."""
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(4096)  # Read first 4KB
                result = chardet.detect(raw)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    async def _run_in_executor(self, func):
        """Run function in thread pool."""
        return await asyncio.get_running_loop().run_in_executor(self._executor, func)
    
    def __del__(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)

# Create a global instance
_extractor = TextExtractor()

# Expose the main methods
async def extract_text(file_path: str) -> str:
    """Extract text from a file."""
    return await _extractor.extract_text(file_path)

def is_supported_file(file_path: str) -> bool:
    """Check if file type is supported."""
    return _extractor.is_supported_file(file_path)