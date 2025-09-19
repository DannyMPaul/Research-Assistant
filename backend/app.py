from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import uuid
from routes.documents import router as documents_router
from routes.search import router as search_router
from routes.vector_search import router as vector_search_router
from routes.chat import router as chat_router

app = FastAPI(title="Document Research Assistant", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Get the root directory (parent of backend)
root_dir = Path(__file__).parent.parent

# Mount static files to serve the frontend
app.mount("/static", StaticFiles(directory=root_dir), name="static")

# Include routers with API prefix
app.include_router(documents_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(vector_search_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML file."""
    return FileResponse(root_dir / "index.html")

@app.get("/api")
async def api_root():
    """API root endpoint providing information."""
    return {"message": "Document Research Assistant API", "version": "4.0.0", "features": ["document_upload", "vector_search", "ai_chat"]}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.docx', '.txt')):
        raise HTTPException(400, "Unsupported file type. Use PDF, DOCX, or TXT.")
    
    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{file_id}{file_extension}"
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "status": "uploaded"
    }

@app.get("/api/documents")
async def list_documents():
    documents = []
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file():
            documents.append({
                "file_id": file_path.stem,
                "filename": file_path.name,
                "size": file_path.stat().st_size
            })
    return {"documents": documents}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
