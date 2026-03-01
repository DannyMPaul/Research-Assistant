"""
FastAPI routes for PageIndex RAG operations.

Endpoints:
  POST   /api/page-index/index/{file_id}   — Index a PDF document
  GET    /api/page-index/status             — List all indexed docs + service status
  GET    /api/page-index/index/{file_id}   — Get index info for one document
  DELETE /api/page-index/index/{file_id}   — Remove an index
  GET    /api/page-index/search             — Reasoning-based search
  POST   /api/page-index/chat              — Chat with PageIndex context

Rate limiting is applied via slowapi decorators (requests / minute per client IP).
"""
import logging
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from services.page_index_service import page_index_service
from services.page_index_retrieval import page_index_retrieval
from services.conversation_manager import get_conversation_manager

logger = logging.getLogger(__name__)

# ---- Rate limiter (keyed by client IP) ----
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/page-index", tags=["PageIndex RAG"])

UPLOAD_DIR = Path("uploads")


# ------------------------------------------------------------------ #
#  Request / Response models                                           #
# ------------------------------------------------------------------ #

class PageIndexChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    file_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these document file IDs. Leave empty to search all indexed PDFs."
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Continue an existing conversation (for multi-turn context)"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sections to retrieve")


class PageIndexChatResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    conversation_id: str
    tokens_used: Optional[int] = None
    retrieval_mode: str = "page_index"


# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

def _resolve_pdf_path(file_id: str) -> Optional[Path]:
    """Find the uploaded file on disk (PDF only). Returns None if not found."""
    matches = list(UPLOAD_DIR.glob(f"{file_id}.pdf"))
    return matches[0] if matches else None


# ------------------------------------------------------------------ #
#  Routes                                                              #
# ------------------------------------------------------------------ #

@router.get("/status")
async def get_pageindex_status():
    """Return PageIndex service availability and list of indexed documents."""
    indexed = page_index_service.list_indexed()
    return {
        "service_available": page_index_service.available,
        "openai_key_configured": bool(settings.OPENAI_API_KEY),
        "model": settings.PAGEINDEX_MODEL,
        "max_concurrent_calls": settings.PAGEINDEX_MAX_CONCURRENT,
        "total_indexed": len(indexed),
        "indexed_documents": indexed,
    }


@router.post("/index/{file_id}")
@limiter.limit(settings.RATE_LIMIT_INDEX)
async def index_document(file_id: str, request: Request):
    """
    Build a PageIndex tree for an uploaded PDF document.

    - Only PDF files can be indexed (PageIndex limitation).
    - Indexing makes multiple OpenAI API calls and may take 1–5 minutes for large documents.
    - The resulting tree JSON is saved on the server for fast subsequent retrieval.
    """
    if not page_index_service.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "PageIndex is not available. Make sure OPENAI_API_KEY is set in .env "
                "and the pageindex package is installed."
            ),
        )

    # Check if already indexed
    if page_index_service.document_indexed(file_id):
        index_info = page_index_service.get_index(file_id)
        return {
            "status": "already_indexed",
            "file_id": file_id,
            "doc_name": index_info.get("doc_name", file_id) if index_info else file_id,
            "message": "Document is already indexed. Use DELETE to re-index.",
        }

    # Verify the file exists and is a PDF
    pdf_path = _resolve_pdf_path(file_id)
    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"PDF file with ID '{file_id}' not found. "
                "Only PDF documents can be indexed with PageIndex."
            ),
        )

    try:
        result = await page_index_service.index_document(file_id, str(pdf_path))
        logger.info(f"Successfully indexed document {file_id}")
        return {
            "status": "indexed",
            **result,
            "message": f"Document indexed successfully with {result['node_count']} sections.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Indexing error for {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/index/{file_id}")
async def get_index_info(file_id: str):
    """Get indexing status and metadata for a specific document."""
    if not page_index_service.document_indexed(file_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{file_id}' has not been indexed yet.",
        )
    tree = page_index_service.get_index(file_id)
    if tree is None:
        raise HTTPException(status_code=500, detail="Failed to load index data.")

    return {
        "file_id": file_id,
        "doc_name": tree.get("doc_name", file_id),
        "doc_description": tree.get("doc_description", ""),
        "node_count": page_index_service._count_nodes(tree.get("structure", [])),
        "top_level_sections": [
            {
                "title": n.get("title", ""),
                "node_id": n.get("node_id", ""),
                "page_range": (
                    f"{n.get('start_index', '')}–{n.get('end_index', '')}"
                    if n.get("start_index") else ""
                ),
                "summary": n.get("summary", ""),
            }
            for n in tree.get("structure", [])
        ],
    }


@router.delete("/index/{file_id}")
async def delete_index(file_id: str):
    """Remove the PageIndex tree for a document (allows re-indexing)."""
    deleted = page_index_service.delete_index(file_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for document '{file_id}'.",
        )
    return {"status": "deleted", "file_id": file_id}


@router.get("/search")
@limiter.limit(settings.RATE_LIMIT_SEARCH)
async def page_index_search(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    file_id: Optional[str] = Query(None, description="Restrict to a specific document"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Reasoning-based search using the PageIndex tree.

    The LLM traverses the document's table-of-contents tree to identify the
    most relevant sections — no vector similarity, pure reasoning.
    """
    if not page_index_retrieval.available:
        raise HTTPException(
            status_code=503,
            detail="PageIndex retrieval unavailable. Check OPENAI_API_KEY.",
        )

    # Select which documents to search
    if file_id:
        target_ids = [file_id]
    else:
        target_ids = [doc["file_id"] for doc in page_index_service.list_indexed()]

    if not target_ids:
        return {
            "query": q,
            "total_results": 0,
            "results": [],
            "message": "No indexed documents found. Index a PDF first.",
        }

    all_results = []
    for fid in target_ids:
        tree = page_index_service.get_index(fid)
        if tree is None:
            continue
        try:
            sections = await page_index_retrieval.search(q, tree, top_k=top_k)
            for s in sections:
                s["file_id"] = fid
            all_results.extend(sections)
        except Exception as e:
            logger.warning(f"Search failed for document {fid}: {e}")

    # Sort globally and limit
    all_results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    all_results = all_results[:top_k]

    return {
        "query": q,
        "total_results": len(all_results),
        "results": all_results,
        "retrieval_mode": "page_index",
    }


@router.post("/chat", response_model=PageIndexChatResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def page_index_chat(body: PageIndexChatRequest, request: Request):
    """
    Ask a question and get an answer powered by PageIndex retrieval.

    Uses reasoning-based tree traversal to find relevant sections, then
    generates a cited answer with OpenAI GPT.
    """
    if not page_index_retrieval.available:
        raise HTTPException(
            status_code=503,
            detail="PageIndex chat unavailable. Check OPENAI_API_KEY.",
        )

    # Select which documents to search
    if body.file_ids:
        target_ids = body.file_ids
    else:
        target_ids = [doc["file_id"] for doc in page_index_service.list_indexed()]

    if not target_ids:
        raise HTTPException(
            status_code=404,
            detail="No indexed PDF documents found. Index at least one PDF first.",
        )

    # Retrieve relevant sections from all target documents
    all_sections = []
    for fid in target_ids:
        tree = page_index_service.get_index(fid)
        if tree is None:
            continue
        try:
            sections = await page_index_retrieval.search(
                body.question, tree, top_k=body.top_k
            )
            for s in sections:
                s["file_id"] = fid
            all_sections.extend(sections)
        except Exception as e:
            logger.warning(f"Retrieval failed for document {fid}: {e}")

    if not all_sections:
        raise HTTPException(
            status_code=404,
            detail="No relevant sections found in the indexed documents.",
        )

    # Sort and take top-k globally
    all_sections.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
    all_sections = all_sections[: body.top_k]

    # Get or create conversation
    manager = await get_conversation_manager()
    conv_id = body.conversation_id
    history = []
    if conv_id:
        history = await manager.get_conversation_history(conv_id, limit=5)
    else:
        conv_id = await manager.create_conversation()

    # Generate answer using OpenAI GPT with retrieved sections as context
    result = await page_index_retrieval.generate_answer(
        question=body.question,
        context_sections=all_sections,
        conversation_history=history,
    )

    # Persist to conversation history
    await manager.add_message(
        conv_id,
        body.question,
        result["answer"],
        result["sources"],
        result["confidence"],
        metadata={"retrieval_mode": "page_index"},
    )

    return PageIndexChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        conversation_id=conv_id,
        tokens_used=result.get("tokens_used"),
        retrieval_mode="page_index",
    )
