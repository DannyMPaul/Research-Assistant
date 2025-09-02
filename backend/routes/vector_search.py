from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extractor import extract_text
from services.vector_store import VectorStore

router = APIRouter()
UPLOAD_DIR = Path("uploads")
vector_store = VectorStore()

@router.post("/document/{file_id}/embed")
async def embed_document(file_id: str):
    if not vector_store.enabled:
        raise HTTPException(503, "Vector embeddings service not available. Install sentence-transformers and faiss-cpu.")
    
    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    
    if not file_paths:
        raise HTTPException(404, "Document not found")
    
    if vector_store.document_exists(file_id):
        return {"message": "Document already embedded", "file_id": file_id}
    
    file_path = file_paths[0]
    
    try:
        text = extract_text(str(file_path))
        chunk_count = vector_store.add_document(file_id, file_path.name, text)
        
        return {
            "file_id": file_id,
            "filename": file_path.name,
            "chunks_created": chunk_count,
            "status": "embedded"
        }
    except Exception as e:
        raise HTTPException(500, f"Error embedding document: {str(e)}")

@router.get("/search/semantic")
async def semantic_search(q: str = Query(..., min_length=1), limit: int = Query(5, ge=1, le=20)):
    if not vector_store.enabled:
        raise HTTPException(503, "Vector embeddings service not available. Install sentence-transformers and faiss-cpu.")
    
    try:
        results = vector_store.search(q, top_k=limit)
        
        return {
            "query": q,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(500, f"Error performing semantic search: {str(e)}")

@router.get("/document/{file_id}/search/semantic")
async def semantic_search_document(file_id: str, q: str = Query(..., min_length=1), limit: int = Query(3, ge=1, le=10)):
    if not vector_store.enabled:
        raise HTTPException(503, "Vector embeddings service not available. Install sentence-transformers and faiss-cpu.")
    
    if not vector_store.document_exists(file_id):
        raise HTTPException(404, "Document not embedded. Please embed it first.")
    
    try:
        results = vector_store.search_by_document(file_id, q, top_k=limit)
        
        return {
            "file_id": file_id,
            "query": q,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(500, f"Error performing document search: {str(e)}")

@router.get("/embeddings/status")
async def get_embeddings_status():
    embedded_docs = []
    total_chunks = 0
    
    for doc in vector_store.documents:
        embedded_docs.append({
            "file_id": doc['file_id'],
            "filename": doc['filename'],
            "chunk_count": doc['chunk_count']
        })
        total_chunks += doc['chunk_count']
    
    return {
        "total_documents": len(embedded_docs),
        "total_chunks": total_chunks,
        "embedded_documents": embedded_docs
    }
