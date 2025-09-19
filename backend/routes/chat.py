from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm_service import llm_service
from services.conversation_manager import conversation_manager
from services.vector_store import VectorStore

router = APIRouter()
vector_store = VectorStore()

class ChatRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    context_limit: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    conversation_id: str
    tokens_used: Optional[int] = None

class ConversationCreate(BaseModel):
    document_ids: Optional[List[str]] = None

@router.post("/chat/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    # Create conversation if none provided
    if not request.conversation_id:
        request.conversation_id = conversation_manager.create_conversation(request.document_ids)
    
    # Verify conversation exists
    conversation = conversation_manager.get_conversation(request.conversation_id)
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    
    try:
        # Get context from vector search
        if request.document_ids:
            # Search within specific documents
            context_chunks = []
            for doc_id in request.document_ids:
                if vector_store.document_exists(doc_id):
                    doc_results = vector_store.search_by_document(
                        doc_id, request.question, top_k=request.context_limit // len(request.document_ids) + 1
                    )
                    context_chunks.extend(doc_results)
        else:
            # Search across all documents
            context_chunks = vector_store.search(request.question, top_k=request.context_limit)
        
        # Get conversation history for context
        conversation_history = conversation_manager.get_conversation_history(
            request.conversation_id, limit=5
        )
        
        # Generate answer using LLM
        llm_response = await llm_service.generate_answer(
            request.question, 
            context_chunks, 
            conversation_history
        )
        
        # Save to conversation
        conversation_manager.add_message(
            request.conversation_id,
            request.question,
            llm_response["answer"],
            llm_response["sources"],
            llm_response["confidence"]
        )
        
        return ChatResponse(
            answer=llm_response["answer"],
            sources=llm_response["sources"],
            confidence=llm_response["confidence"],
            conversation_id=request.conversation_id,
            tokens_used=llm_response.get("tokens_used")
        )
    
    except Exception as e:
        raise HTTPException(500, f"Error processing question: {str(e)}")

@router.post("/chat/conversations")
async def create_conversation(request: ConversationCreate):
    conversation_id = conversation_manager.create_conversation(request.document_ids)
    return {
        "conversation_id": conversation_id,
        "document_ids": request.document_ids or [],
        "status": "created"
    }

@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, limit: int = Query(10, ge=1, le=50)):
    conversation = conversation_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(404, "Conversation not found")
    
    messages = conversation_manager.get_conversation_history(conversation_id, limit)
    
    return {
        "conversation_id": conversation_id,
        "created_at": conversation["created_at"],
        "document_ids": conversation.get("document_ids", []),
        "total_messages": conversation["metadata"]["total_questions"],
        "messages": messages
    }

@router.get("/chat/conversations")
async def list_conversations(limit: int = Query(20, ge=1, le=100)):
    conversations = conversation_manager.list_conversations(limit)
    return {
        "conversations": conversations,
        "total": len(conversations)
    }

@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(404, "Conversation not found")
    
    return {"status": "deleted", "conversation_id": conversation_id}

@router.get("/chat/status")
async def get_chat_status():
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
    generate_answer: bool = Query(False),
    document_ids: Optional[List[str]] = Query(None),
    limit: int = Query(5, ge=1, le=20)
):
    try:
        # Get search results
        if document_ids:
            search_results = []
            for doc_id in document_ids:
                if vector_store.document_exists(doc_id):
                    doc_results = vector_store.search_by_document(doc_id, q, top_k=limit // len(document_ids) + 1)
                    search_results.extend(doc_results)
        else:
            search_results = vector_store.search(q, top_k=limit)
        
        response = {
            "query": q,
            "results": search_results,
            "total_results": len(search_results)
        }
        
        # Generate AI answer if requested
        if generate_answer and llm_service.enabled:
            llm_response = await llm_service.generate_answer(q, search_results)
            response["ai_answer"] = {
                "answer": llm_response["answer"],
                "confidence": llm_response["confidence"],
                "sources": llm_response["sources"]
            }
        
        return response
    
    except Exception as e:
        raise HTTPException(500, f"Error performing enhanced search: {str(e)}")