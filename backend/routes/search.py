from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extractor import extract_text
from utils.text_processor import chunk_text, search_in_text

router = APIRouter()
UPLOAD_DIR = Path("uploads")

@router.get("/document/{file_id}/text")
async def get_document_text(file_id: str):
    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    
    if not file_paths:
        raise HTTPException(404, "Document not found")
    
    file_path = file_paths[0]
    
    try:
        text = extract_text(str(file_path))
        return {
            "file_id": file_id,
            "filename": file_path.name,
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    except Exception as e:
        raise HTTPException(500, f"Error extracting text: {str(e)}")

@router.get("/document/{file_id}/chunks")
async def get_document_chunks(file_id: str, chunk_size: int = Query(500, ge=100, le=2000)):
    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    
    if not file_paths:
        raise HTTPException(404, "Document not found")
    
    file_path = file_paths[0]
    
    try:
        text = extract_text(str(file_path))
        chunks = chunk_text(text, chunk_size=chunk_size)
        
        return {
            "file_id": file_id,
            "filename": file_path.name,
            "total_chunks": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(500, f"Error processing text: {str(e)}")

@router.get("/document/{file_id}/search")
async def search_document(file_id: str, q: str = Query(..., min_length=1)):
    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    
    if not file_paths:
        raise HTTPException(404, "Document not found")
    
    file_path = file_paths[0]
    
    try:
        text = extract_text(str(file_path))
        results = search_in_text(text, q)
        
        return {
            "file_id": file_id,
            "filename": file_path.name,
            "query": q,
            "total_matches": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(500, f"Error searching document: {str(e)}")

@router.get("/search")
async def search_all_documents(q: str = Query(..., min_length=1)):
    all_results = []
    
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            try:
                text = extract_text(str(file_path))
                results = search_in_text(text, q)
                
                if results:
                    all_results.append({
                        "file_id": file_path.stem,
                        "filename": file_path.name,
                        "matches": len(results),
                        "results": results[:3]  # Limit to top 3 matches per document
                    })
            except Exception:
                continue
    
    return {
        "query": q,
        "total_documents": len(all_results),
        "documents": all_results
    }
