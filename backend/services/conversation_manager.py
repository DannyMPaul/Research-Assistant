import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import aiofiles
import aiofiles.os
from fastapi.encoders import jsonable_encoder
from collections import OrderedDict
import time
from utils.error_handling import DatabaseError, NotFoundError

logger = logging.getLogger(__name__)

class ConversationError(Exception):
    """Base class for conversation-related errors."""
    pass

class ConversationTimeoutError(ConversationError):
    """Raised when a conversation operation times out."""
    pass

class ConversationNotFoundError(ConversationError):
    """Raised when a requested conversation is not found."""
    pass

class TimeoutError(Exception):
    """Custom timeout error."""
    pass

class ConversationCache:
    """LRU cache for conversation data."""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict]:
        """Get item from cache with async lock and timeout."""
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    if key in self.cache:
                        value = self.cache.pop(key)
                        self.cache[key] = value
                        return value
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Cache get operation timed out for key: {key}")
            return None
    
    async def put(self, key: str, value: Dict, timeout: float = 2.0):
        """Put item in cache with async lock and timeout."""
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    if key in self.cache:
                        self.cache.pop(key)
                    elif len(self.cache) >= self.max_size:
                        self.cache.popitem(last=False)
                    self.cache[key] = value
        except asyncio.TimeoutError:
            logger.warning(f"Cache put operation timed out for key: {key}")

    async def remove(self, key: str, timeout: float = 2.0):
        """Remove item from cache with async lock and timeout."""
        try:
            async with asyncio.timeout(timeout):
                async with self._lock:
                    self.cache.pop(key, None)
        except asyncio.TimeoutError:
            logger.warning(f"Cache remove operation timed out for key: {key}")

class ConversationManager:
    def __init__(self, cache_size: int = 100):
        self.conversations_dir = Path("conversations")
        self.conversations_dir.mkdir(exist_ok=True)
        self.cache = ConversationCache(max_size=cache_size)
        self._saving: Set[str] = set()
        self._save_lock = asyncio.Lock()
        self._cleanup_task = None
    
    async def start(self):
        """Initialize the conversation manager."""
        await self._start_cleanup_task()
    
    async def _start_cleanup_task(self):
        """Start the periodic cleanup task with error handling."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            self._cleanup_task.add_done_callback(self._handle_cleanup_task_done)
    
    def _handle_cleanup_task_done(self, task):
        """Handle cleanup task completion and restart if needed."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}")
            # Restart the task after a delay
            asyncio.create_task(self._delayed_cleanup_restart())
    
    async def _delayed_cleanup_restart(self):
        """Restart cleanup task after a delay."""
        await asyncio.sleep(60)  # Wait 1 minute before restarting
        self._start_cleanup_task()
    
    async def create_conversation(
        self, 
        document_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        timeout: float = 5.0
    ) -> str:
        """Create a new conversation with optimized storage and timeout handling."""
        conversation_id = str(uuid.uuid4())
        conversation = {
            "id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "document_ids": document_ids or [],
            "messages": [],
            "metadata": {
                "total_questions": 0,
                "last_activity": datetime.now().isoformat(),
                **(metadata or {})
            }
        }
        
        try:
            async with asyncio.timeout(timeout):
                await self.cache.put(conversation_id, conversation)
                await self._save_conversation(conversation)
                
            logger.info(f"Created conversation {conversation_id}")
            return conversation_id
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout creating conversation {conversation_id}")
            raise ConversationTimeoutError("Failed to create conversation due to timeout")
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise DatabaseError("Failed to create conversation")
    
    async def add_message(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        sources: Optional[List[Dict]] = None,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add message to conversation with optimized updates."""
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                raise NotFoundError(f"Conversation {conversation_id} not found")
            
            message = {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "sources": sources or [],
                "confidence": confidence,
                "metadata": metadata or {}
            }
            
            # Update conversation in memory first
            conversation["messages"].append(message)
            conversation["metadata"].update({
                "total_questions": conversation["metadata"]["total_questions"] + 1,
                "last_activity": datetime.now().isoformat()
            })
            
            # Update cache and persist
            await self.cache.put(conversation_id, conversation)
            await self._save_conversation(conversation)
            
            logger.info(f"Added message to conversation {conversation_id}")
            return True
        
        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            raise DatabaseError("Failed to add message to conversation")
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation with caching."""
        try:
            # Try cache first
            conversation = await self.cache.get(conversation_id)
            if conversation:
                return conversation
            
            # Load from disk if not in cache
            return await self._load_conversation(conversation_id)
        
        except Exception as e:
            logger.error(f"Failed to get conversation {conversation_id}: {e}")
            return None
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 10,
        before_timestamp: Optional[str] = None
    ) -> List[Dict]:
        """Get conversation history with pagination."""
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                return []
            
            messages = conversation.get("messages", [])
            
            if before_timestamp:
                messages = [
                    m for m in messages 
                    if m["timestamp"] < before_timestamp
                ]
            
            return messages[-limit:] if limit > 0 else messages
        
        except Exception as e:
            logger.error(f"Failed to get conversation history {conversation_id}: {e}")
            return []
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation with cleanup."""
        try:
            # Remove from cache
            await self.cache.remove(conversation_id)
            
            # Remove file
            conversation_file = self.conversations_dir / f"{conversation_id}.json"
            if await aiofiles.os.path.exists(conversation_file):
                await aiofiles.os.remove(conversation_file)
                logger.info(f"Deleted conversation {conversation_id}")
                return True
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            return False
    
    async def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
        filter_criteria: Optional[Dict] = None
    ) -> List[Dict]:
        """List conversations with filtering and pagination."""
        try:
            conversations = []
            
            # Scan conversation files
            async for conv_file in self._scan_conversation_files():
                try:
                    conversation_id = conv_file.stem
                    conversation = await self.get_conversation(conversation_id)
                    
                    if conversation and self._matches_filter(conversation, filter_criteria):
                        conversations.append(
                            await self._get_conversation_summary(conversation)
                        )
                except Exception as e:
                    logger.warning(f"Failed to load conversation {conv_file}: {e}")
                    continue
            
            # Sort by last activity
            conversations.sort(
                key=lambda x: x["last_activity"],
                reverse=True
            )
            
            return conversations[offset:offset + limit]
        
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return []
    
    async def cleanup_old_conversations(self, days: int = 7):
        """Clean up old conversations asynchronously."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            async for conv_file in self._scan_conversation_files():
                try:
                    stat = await aiofiles.os.stat(conv_file)
                    if datetime.fromtimestamp(stat.st_mtime) < cutoff_time:
                        conversation_id = conv_file.stem
                        await self.delete_conversation(conversation_id)
                except Exception as e:
                    logger.warning(f"Failed to cleanup conversation {conv_file}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to cleanup conversations: {e}")
    
    @asynccontextmanager
    async def _file_lock(self, conversation_id: str):
        """Ensure atomic file operations."""
        async with self._save_lock:
            while conversation_id in self._saving:
                await asyncio.sleep(0.1)
            self._saving.add(conversation_id)
        try:
            yield
        finally:
            async with self._save_lock:
                self._saving.remove(conversation_id)
    
    async def _save_conversation(self, conversation: Dict):
        """Save conversation with atomic file operations."""
        conversation_id = conversation["id"]
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        
        async with self._file_lock(conversation_id):
            try:
                async with aiofiles.open(conversation_file, 'w', encoding='utf-8') as f:
                    await f.write(
                        json.dumps(
                            jsonable_encoder(conversation),
                            indent=2,
                            ensure_ascii=False
                        )
                    )
            except Exception as e:
                logger.error(f"Failed to save conversation {conversation_id}: {e}")
                raise DatabaseError(f"Failed to save conversation {conversation_id}")
    
    async def _load_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Load conversation from disk."""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        
        if not await aiofiles.os.path.exists(conversation_file):
            return None
        
        try:
            async with aiofiles.open(conversation_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                conversation = json.loads(content)
                await self.cache.put(conversation_id, conversation)
                return conversation
        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return None
    
    async def _get_conversation_summary(self, conversation: Dict) -> Dict:
        """Generate conversation summary."""
        messages = conversation.get("messages", [])
        last_message = messages[-1] if messages else None
        
        return {
            "id": conversation["id"],
            "created_at": conversation["created_at"],
            "last_activity": conversation["metadata"]["last_activity"],
            "total_questions": conversation["metadata"]["total_questions"],
            "document_ids": conversation.get("document_ids", []),
            "last_question": (
                f"{last_message['question'][:100]}..."
                if last_message and len(last_message["question"]) > 100
                else last_message["question"] if last_message else None
            ),
            "last_answer_preview": (
                f"{last_message['answer'][:150]}..."
                if last_message and len(last_message["answer"]) > 150
                else last_message["answer"] if last_message else None
            ),
            "metadata": conversation.get("metadata", {})
        }
    
    async def _scan_conversation_files(self):
        """Async generator for scanning conversation files."""
        for conv_file in self.conversations_dir.glob("*.json"):
            yield conv_file
    
    def _matches_filter(self, conversation: Dict, filter_criteria: Optional[Dict]) -> bool:
        """Check if conversation matches filter criteria."""
        if not filter_criteria:
            return True
        
        for key, value in filter_criteria.items():
            if key == "document_ids":
                if not any(doc_id in conversation.get("document_ids", []) 
                          for doc_id in value):
                    return False
            elif key == "date_range":
                created_at = datetime.fromisoformat(conversation["created_at"])
                if not (value["start"] <= created_at <= value["end"]):
                    return False
            elif key in conversation.get("metadata", {}):
                if conversation["metadata"][key] != value:
                    return False
        return True
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(24 * 60 * 60)  # Run daily
                await self.cleanup_old_conversations()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error

conversation_manager = None
_manager_lock = asyncio.Lock()

async def get_conversation_manager():
    global conversation_manager
    if conversation_manager is not None:
        return conversation_manager
    async with _manager_lock:
        if conversation_manager is None:
            conversation_manager = ConversationManager()
            await conversation_manager.start()
    return conversation_manager