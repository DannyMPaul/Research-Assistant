from fastapi import APIRouter, HTTPException, Query, Depends, Request
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import asyncio
from datetime import datetime, timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import settings
from datetime import datetime, timedelta

from utils.text_extractor import TextExtractor
from utils.text_processor import TextProcessor
from utils.error_handling import NotFoundError, AppError

logger = logging.getLogger(__name__)
router = APIRouter()

class DocumentService:
    """Handles document operations with caching."""
    def __init__(self):
        self.upload_dir = Path(__file__).parent.parent / "uploads"
        self.text_extractor = TextExtractor()
        self.text_processor = TextProcessor()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()
        self._last_cleanup = datetime.now()
        
    async def get_document_path(self, file_id: str) -> Path:
        """Get document path with validation."""
        paths = list(self.upload_dir.glob(f"{file_id}.*"))
        if not paths:
            raise NotFoundError("Document not found")
        return paths[0]
    
    async def get_document_text(self, file_id: str) -> Dict[str, Any]:
        async with self._cache_lock:
            if file_id in self._cache:
                return self._cache[file_id]
        file_path = await self.get_document_path(file_id)
        try:
            text = await self.text_extractor.extract_text(str(file_path))
            result = {
                "file_id": file_id,
                "filename": file_path.name,
                "text": text,
                "word_count": len(text.split()),
                "char_count": len(text),
                "last_modified": file_path.stat().st_mtime,
            }
            async with self._cache_lock:
                self._cache[file_id] = result
            return result
        except Exception as e:
            logger.error(f"Text extraction error for {file_id}: {e}")
            raise AppError("Failed to extract text.")
    
    async def get_document_chunks(
        self,
        file_id: str,
        chunk_size: int = 500
    ) -> Dict[str, Any]:
        """Get document chunks with efficient processing."""
        doc = await self.get_document_text(file_id)
        chunks = self.text_processor.chunk_text(doc["text"], chunk_size=chunk_size)
        
        return {
            "file_id": file_id,
            "filename": doc["filename"],
            "total_chunks": len(chunks),
            "chunks": chunks
        }
    
    async def search_document(
        self,
        file_id: str,
        query: str,
        context_length: int = 200
    ) -> Dict[str, Any]:
        """Search within a specific document."""
        doc = await self.get_document_text(file_id)
        results = self.text_processor.search_in_text(
            doc["text"],
            query,
            context_length=context_length
        )
        
        return {
            "file_id": file_id,
            "filename": doc["filename"],
            "query": query,
            "total_matches": len(results),
            "results": results
        }
    
    async def search_all_documents(
        self,
        query: str,
        max_results_per_doc: int = 3
    ) -> Dict[str, Any]:
        """Search across all documents efficiently."""
        all_results = []
        tasks = []
        
        async for file_path in self._scan_documents():
            tasks.append(self._search_single_document(
                file_path,
                query,
                max_results_per_doc
            ))
        
        # Execute searches in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results = [r for r in results if r is not None]
        
        return {
            "query": query,
            "total_documents": len(all_results),
            "documents": all_results
        }
    
    async def _search_single_document(
        self,
        file_path: Path,
        query: str,
        max_results: int
    ) -> Optional[Dict[str, Any]]:
        """Search a single document with error handling."""
        try:
            text = await self.text_extractor.extract_text(str(file_path))
            results = self.text_processor.search_in_text(text, query)
            
            if results:
                return {
                    "file_id": file_path.stem,
                    "filename": file_path.name,
                    "matches": len(results),
                    "results": results[:max_results]
                }
        except Exception as e:
            logger.warning(f"Error searching {file_path}: {e}")
        return None
    
    async def _scan_documents(self):
        """Async generator for scanning documents."""
        for file_path in self.upload_dir.glob("*"):
            if file_path.is_file():
                yield file_path

# Initialize service
document_service = DocumentService()

# Dependency for common parameters
class SearchParams:
    def __init__(
        self,
        context_length: int = Query(200, ge=50, le=500),
        chunk_size: int = Query(500, ge=100, le=2000)
    ):
        self.context_length = context_length
        self.chunk_size = chunk_size

async def get_search_params(
    context_length: int = Query(200, ge=50, le=500),
    chunk_size: int = Query(500, ge=100, le=2000)
) -> SearchParams:
    return SearchParams(context_length, chunk_size)

# Routes
@router.get("/document/{file_id}/text")
async def get_document_text(file_id: str):
    """Get full text content of a document."""
    try:
        return await document_service.get_document_text(file_id)
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document text: {e}")
        raise HTTPException(500, "Failed to retrieve document text")

@router.get("/document/{file_id}/chunks")
async def get_document_chunks(
    file_id: str,
    params: SearchParams = Depends(get_search_params)
):
    """Get document split into chunks."""
    try:
        return await document_service.get_document_chunks(
            file_id,
            chunk_size=params.chunk_size
        )
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document chunks: {e}")
        raise HTTPException(500, "Failed to retrieve document chunks")

@router.get("/document/{file_id}/search")
async def search_document(
    file_id: str,
    q: str = Query(..., min_length=1),
    params: SearchParams = Depends(get_search_params)
):
    """Search within a specific document."""
    try:
        return await document_service.search_document(
            file_id,
            q,
            context_length=params.context_length
        )
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error searching document: {e}")
        raise HTTPException(500, "Search operation failed")

@router.get("/search")
async def search_all_documents(
    q: str = Query(..., min_length=1),
    max_results_per_doc: int = Query(3, ge=1, le=10)
):
    """Search across all documents."""
    try:
        return await document_service.search_all_documents(
            q,
            max_results_per_doc=max_results_per_doc
        )
    except Exception as e:
        logger.error(f"Error searching all documents: {e}")
        raise HTTPException(500, "Search operation failed")