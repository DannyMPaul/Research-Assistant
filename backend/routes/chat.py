from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from fastapi.responses import JSONResponse

from services.llm_service import llm_service
from services.conversation_manager import get_conversation_manager
from services.dependencies import vector_store

async def get_manager():
    return await get_conversation_manager()

manager_dependency = Depends(get_manager)
from utils.text_extractor import extract_text
from utils.text_processor import TextProcessor
from utils.error_handling import AppError, NotFoundError

text_processor = TextProcessor()

logger = logging.getLogger(__name__)
router = APIRouter()

# Models
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    context_limit: int = Field(default=5, ge=1, le=20)

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    conversation_id: str
    tokens_used: Optional[int] = None

class ConversationCreate(BaseModel):
    document_ids: Optional[List[str]] = None

# Dependency for conversation validation
async def get_conversation(conversation_id: str, manager = manager_dependency) -> Dict[str, Any]:
    """Validate and return conversation."""
    conversation = await manager.get_conversation(conversation_id)
    if not conversation:
        raise NotFoundError("Conversation not found")
    return conversation

class SearchService:
    def __init__(self, vs):
        self.vector_store = vs
        self.upload_dir = Path(__file__).parent.parent / "uploads"

    async def search_documents(
        self, 
        query: str,
        document_ids: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform vector search with fallback to keyword search."""
        if document_ids:
            results = []
            per_doc_limit = limit // len(document_ids) + 1
            for doc_id in document_ids:
                if self.vector_store.document_exists(doc_id):
                    doc_results = self.vector_store.search_by_document(
                        doc_id, query, top_k=per_doc_limit
                    )
                    results.extend(doc_results)
            return results[:limit]

        results = await self.vector_store.search(query, top_k=limit)
        if not results:
            return await self._keyword_search(query, limit, document_ids)
        return results

    async def _keyword_search(
        self,
        query: str,
        limit: int = 5,
        allowed_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback keyword-based search."""
        results = []
        try:
            for file_path in self.upload_dir.glob("*"):
                if not file_path.is_file():
                    continue
                    
                file_id = file_path.stem
                if allowed_ids and file_id not in allowed_ids:
                    continue
                
                try:
                    text = await extract_text(str(file_path))
                    matches = text_processor.search_in_text(text, query, context_length=200)
                    if matches:
                        score = min(1.0, 0.2 + 0.1 * len(matches))
                        results.append({
                            "file_id": file_id,
                            "filename": file_path.name,
                            "chunk_text": matches[0]["context"],
                            "similarity_score": score,
                        })
                except Exception as e:
                    logger.warning(f"Error searching file {file_path}: {e}")
                    continue
                    
            results.sort(key=lambda r: r["similarity_score"], reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

search_service = SearchService(vector_store)

# Routes
@router.post("/chat/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest, manager = manager_dependency):
    """Process chat questions with context from document search."""
    try:
        # Create or validate conversation
        conv_id = request.conversation_id or await manager.create_conversation(request.document_ids)
        conversation = await get_conversation(conv_id, manager)
        
        # Get search results for context
        context_chunks = await search_service.search_documents(
            request.question,
            request.document_ids,
            request.context_limit
        )
        
        # Get conversation history
        history = await manager.get_conversation_history(conv_id, limit=5)
        
        # Generate answer
        response = await llm_service.generate_answer(
            request.question,
            context_chunks,
            history
        )
        
        # Save to conversation
        await manager.add_message(
            conv_id,
            request.question,
            response["answer"],
            response["sources"],
            response["confidence"]
        )
        
        return ChatResponse(
            answer=response["answer"],
            sources=response["sources"],
            confidence=response["confidence"],
            conversation_id=conv_id,
            tokens_used=response.get("tokens_used")
        )
        
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(500, "Failed to process question")

@router.post("/chat/conversations")
async def create_conversation(request: ConversationCreate, manager = manager_dependency):
    """Create a new conversation."""
    conversation_id = await manager.create_conversation(request.document_ids)
    return {
        "conversation_id": conversation_id,
        "document_ids": request.document_ids or [],
        "status": "created"
    }

@router.get("/chat/conversations/{conversation_id}")
async def get_conversation_details(
    conversation: Dict[str, Any] = Depends(get_conversation),
    limit: int = Query(10, ge=1, le=50),
    manager = manager_dependency
):
    """Get conversation details and messages."""
    messages = await manager.get_conversation_history(
        conversation["id"],
        limit=limit
    )
    
    return {
        "conversation_id": conversation["id"],
        "created_at": conversation["created_at"],
        "document_ids": conversation.get("document_ids", []),
        "total_messages": conversation["metadata"]["total_questions"],
        "messages": messages
    }

@router.get("/chat/conversations")
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    manager = manager_dependency
):
    """List all conversations with pagination."""
    conversations = await manager.list_conversations(
        limit=limit,
        offset=offset
    )
    return {
        "conversations": conversations,
        "total": len(conversations)
    }

@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, manager = manager_dependency):
    """Delete a conversation."""
    if await manager.delete_conversation(conversation_id):
        return {"status": "deleted", "conversation_id": conversation_id}
    raise NotFoundError("Conversation not found")

@router.get("/chat/status")
async def get_chat_status():
    """Get current status of chat services."""
    return {
        "llm_available": llm_service.enabled,
        "model_name": llm_service.model_name if llm_service.enabled else None,
        "vector_search_available": vector_store.enabled,
        "total_documents": len(vector_store.documents),
        "total_chunks": len(vector_store.chunks)
    }

@router.get("/search/enhanced")
async def enhanced_search(
    q: str = Query(..., min_length=1),
    generate_answer: bool = False,
    document_ids: Optional[List[str]] = Query(None),
    limit: int = Query(5, ge=1, le=20)
):
    """Enhanced search with optional AI-generated answers."""
    try:
        # Get search results
        results = await search_service.search_documents(q, document_ids, limit)
        
        response = {
            "query": q,
            "results": results,
            "total_results": len(results)
        }
        
        # Generate AI answer if requested
        if generate_answer and llm_service.enabled:
            try:
                llm_response = await llm_service.generate_answer(q, results)
                if isinstance(llm_response, dict):
                    response["ai_answer"] = {
                        "answer": llm_response.get("answer", ""),
                        "confidence": llm_response.get("confidence", 0.0),
                        "sources": llm_response.get("sources", [])
                    }
                else:
                    response["ai_answer"] = {
                        "answer": "Error: Invalid response format from language model",
                        "confidence": 0.0,
                        "sources": []
                    }
            except Exception as e:
                response["ai_answer"] = {
                    "answer": f"Error generating answer: {str(e)}",
                    "confidence": 0.0,
                    "sources": []
                }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        raise HTTPException(500, "Search operation failed")