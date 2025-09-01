from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import uuid
from routes.documents import router as documents_router
from routes.search import router as search_router

app = FastAPI(title="Document Research Assistant", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app.include_router(documents_router, prefix="/api")
app.include_router(search_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Document Research Assistant API", "version": "0.2.0"}

@app.post("/upload")
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

@app.get("/documents")
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
