from fastapi import APIRouter, HTTPException
from pathlib import Path
import re
import logging
from utils.text_extractor import extract_text
from services.dependencies import vector_store
from pydantic import BaseModel

router = APIRouter()
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 ._-]{0,253}[a-zA-Z0-9]$")

class RenameRequest(BaseModel):
    new_filename: str

@router.get("/document/{file_id}/text")
async def get_document_text(file_id: str):
    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not file_paths:
        raise HTTPException(404, "Document not found.")
    try:
        text = await extract_text(str(file_paths[0]))
        return {"file_id": file_id, "text": text, "word_count": len(text.split())}
    except Exception as e:
        logging.error(f"Text extraction error for {file_id}: {e}")
        raise HTTPException(500, "Failed to extract document text.")

@router.post("/document/{file_id}/rename")
async def rename_document(file_id: str, payload: RenameRequest):
    file_paths = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not file_paths:
        raise HTTPException(404, "Document not found.")
    current_path = file_paths[0]

    new_name = payload.new_filename.strip()
    if not new_name or not _SAFE_NAME_RE.match(new_name):
        raise HTTPException(400, "Filename contains invalid characters.")

    if "." not in Path(new_name).name:
        new_name = f"{Path(new_name).stem}{current_path.suffix}"

    candidate = (UPLOAD_DIR / new_name).resolve()
    if candidate.parent != UPLOAD_DIR.resolve():
        raise HTTPException(400, "Invalid filename.")

    if candidate.exists():
        raise HTTPException(400, "A file with that name already exists.")

    try:
        current_path.rename(candidate)
        if vector_store.document_exists(file_id):
            vector_store.update_document_filename(file_id, candidate.name)
        return {"file_id": file_id, "old_filename": current_path.name, "new_filename": candidate.name}
    except Exception as e:
        logging.error(f"Rename error for {file_id}: {e}")
        raise HTTPException(500, "Failed to rename document.")
