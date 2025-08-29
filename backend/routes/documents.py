from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_extractor import extract_text

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
            "text": text,
            "word_count": len(text.split())
        }
    except Exception as e:
        raise HTTPException(500, f"Error extracting text: {str(e)}")
