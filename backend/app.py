from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import re
from pathlib import Path
import uuid
import logging
from pythonjsonlogger import jsonlogger
from routes.documents import router as documents_router
from routes.search import router as search_router
from routes.vector_search import router as vector_search_router
from routes.chat import router as chat_router
from routes.page_index import router as page_index_router
from config import settings
from utils.error_handling import setup_error_handlers

try:
    import magic
    _MAGIC_AVAILABLE = True
except ImportError:
    _MAGIC_AVAILABLE = False

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

_MIME_MAP = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt":  "text/",
}

_SAFE_EXTENSIONS = set(settings.ALLOWED_EXTENSIONS)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

limiter = Limiter(key_func=get_remote_address)

docs_url = "/api/docs" if settings.APP_ENV == "development" else None
redoc_url = "/api/redoc" if settings.APP_ENV == "development" else None

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    docs_url=docs_url,
    redoc_url=redoc_url,
)

setup_error_handlers(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.exception(f"Unhandled error: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "An internal server error occurred"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(status_code=422, content={"detail": "Invalid request parameters"})


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

root_dir = Path(__file__).parent.parent
app.mount("/static", StaticFiles(directory=root_dir), name="static")

app.include_router(documents_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(vector_search_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(page_index_router, prefix="/api")


@app.get("/")
async def serve_frontend():
    return FileResponse(root_dir / "index.html")


@app.get("/api")
async def api_root():
    return {
        "message": "Document Research Assistant API",
        "version": settings.API_VERSION,
        "features": [
            "document_upload",
            "vector_search",
            "ai_chat",
            "page_index_rag",
            "document_organization",
        ],
    }


def _validate_mime(content: bytes, extension: str) -> bool:
    if not _MAGIC_AVAILABLE:
        return True
    detected = magic.from_buffer(content[:2048], mime=True)
    expected = _MIME_MAP.get(extension, "")
    if not expected:
        return False
    if expected.endswith("/"):
        return detected.startswith(expected)
    return detected == expected


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    raw_suffix = Path(file.filename or "").suffix.lower()
    if raw_suffix not in _SAFE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(_SAFE_EXTENSIONS)}")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max: {settings.MAX_UPLOAD_SIZE // (1024 * 1024)}MB")

    if not _validate_mime(content, raw_suffix):
        raise HTTPException(400, "File content does not match declared file type.")

    existing = sum(1 for f in settings.UPLOAD_DIR.iterdir() if f.is_file())
    if existing >= settings.MAX_TOTAL_FILES:
        raise HTTPException(429, f"Upload limit of {settings.MAX_TOTAL_FILES} files reached.")

    file_id = str(uuid.uuid4())
    file_path = settings.UPLOAD_DIR / f"{file_id}{raw_suffix}"

    try:
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Uploaded: id={file_id} ext={raw_suffix} size={len(content)}")
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "status": "uploaded",
        }
    except Exception as e:
        logger.error(f"Upload write failed: {e}")
        raise HTTPException(500, "Failed to save file.")


@app.delete("/api/upload/{file_id}")
async def delete_document(file_id: str):
    if not _UUID_RE.match(file_id):
        raise HTTPException(400, "Invalid document ID.")

    from services.dependencies import vector_store
    from services.page_index_service import page_index_service

    deleted_file = False
    for file_path in settings.UPLOAD_DIR.glob(f"{file_id}.*"):
        file_path.unlink()
        deleted_file = True

    vector_store.remove_document(file_id)
    page_index_service.delete_index(file_id)

    if not deleted_file:
        raise HTTPException(404, "Document not found.")

    logger.info(f"Deleted document: id={file_id}")
    return {"status": "deleted", "file_id": file_id}


@app.get("/api/documents")
async def list_documents():
    try:
        documents = []
        for file_path in settings.UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in _SAFE_EXTENSIONS:
                documents.append({
                    "file_id": file_path.stem,
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "extension": file_path.suffix.lower(),
                    "last_modified": file_path.stat().st_mtime,
                })
        documents.sort(key=lambda x: x["last_modified"], reverse=True)
        return {
            "documents": documents,
            "total_count": len(documents),
            "total_size": sum(d["size"] for d in documents),
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(500, "Failed to retrieve document list.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.APP_HOST, port=settings.APP_PORT)
