from fastapi import APIRouter, HTTPException, Query, Request
from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import settings
from utils.text_extractor import extract_text
from services.dependencies import vector_store

router = APIRouter()
UPLOAD_DIR = settings.UPLOAD_DIR
limiter = Limiter(key_func=get_remote_address)


@router.post("/document/{file_id}/embed")
@limiter.limit(settings.RATE_LIMIT_INDEX)
async def embed_document(file_id: str, request: Request):
    if not vector_store.enabled:
        raise HTTPException(503, "Vector embeddings not available. Install sentence-transformers and faiss-cpu.")

    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not file_paths:
        raise HTTPException(404, "Document not found.")

    if vector_store.document_exists(file_id):
        return {"message": "Document already embedded.", "file_id": file_id}

    file_path = file_paths[0]
    try:
        text = await extract_text(str(file_path))
        chunk_count = await vector_store.add_document(file_id, file_path.name, text)
        return {"file_id": file_id, "filename": file_path.name, "chunks_created": chunk_count, "status": "embedded"}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Embed error for {file_id}: {e}")
        raise HTTPException(500, "Failed to embed document.")


@router.get("/search/semantic")
@limiter.limit(settings.RATE_LIMIT_SEARCH)
async def semantic_search(request: Request, q: str = Query(..., min_length=1, max_length=500), limit: int = Query(5, ge=1, le=20)):
    if not vector_store.enabled:
        raise HTTPException(503, "Vector embeddings not available. Install sentence-transformers and faiss-cpu.")
    try:
        results = await vector_store.search(q, top_k=limit)
        return {"query": q, "total_results": len(results), "results": results}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Semantic search error: {e}")
        raise HTTPException(500, "Search operation failed.")


@router.get("/document/{file_id}/search/semantic")
@limiter.limit(settings.RATE_LIMIT_SEARCH)
async def semantic_search_document(request: Request, file_id: str, q: str = Query(..., min_length=1, max_length=500), limit: int = Query(3, ge=1, le=10)):
    if not vector_store.enabled:
        raise HTTPException(503, "Vector embeddings not available. Install sentence-transformers and faiss-cpu.")
    if not vector_store.document_exists(file_id):
        raise HTTPException(404, "Document not embedded. Please embed it first.")
    try:
        results = vector_store.search_by_document(file_id, q, top_k=limit)
        return {"file_id": file_id, "query": q, "total_results": len(results), "results": results}
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Document semantic search error: {e}")
        raise HTTPException(500, "Search operation failed.")


@router.get("/embeddings/status")
async def get_embeddings_status():
    embedded_docs = []
    total_chunks = 0
    embedded_file_ids = set()
    for doc in vector_store.documents:
        embedded_docs.append({
            "file_id": doc["file_id"],
            "filename": doc["filename"],
            "chunk_count": doc["chunk_count"],
        })
        total_chunks += doc["chunk_count"]
        embedded_file_ids.add(doc["file_id"])
    return {
        "total_documents": len(embedded_docs),
        "total_chunks": total_chunks,
        "embedded_documents": embedded_docs,
        "embedded_file_ids": list(embedded_file_ids),
    }
