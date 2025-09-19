import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class ConversationManager:
    def __init__(self):
        self.conversations_dir = Path("conversations")
        self.conversations_dir.mkdir(exist_ok=True)
        self.active_conversations = {}
    
    def create_conversation(self, document_ids: List[str] = None) -> str:
        conversation_id = str(uuid.uuid4())
        conversation = {
            "id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "document_ids": document_ids or [],
            "messages": [],
            "metadata": {
                "total_questions": 0,
                "last_activity": datetime.now().isoformat()
            }
        }
        
        self.active_conversations[conversation_id] = conversation
        self._save_conversation(conversation)
        return conversation_id
    
    def add_message(self, conversation_id: str, question: str, answer: str, 
                   sources: List[Dict] = None, confidence: float = 0.0) -> bool:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        message = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "confidence": confidence,
            "message_id": str(uuid.uuid4())
        }
        
        conversation["messages"].append(message)
        conversation["metadata"]["total_questions"] += 1
        conversation["metadata"]["last_activity"] = datetime.now().isoformat()
        
        self.active_conversations[conversation_id] = conversation
        self._save_conversation(conversation)
        return True
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        return self._load_conversation(conversation_id)
    
    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.get("messages", [])
        return messages[-limit:] if limit > 0 else messages
    
    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        if conversation_file.exists():
            conversation_file.unlink()
            return True
        return False
    
    def list_conversations(self, limit: int = 20) -> List[Dict]:
        conversations = []
        
        # Add active conversations
        for conv in self.active_conversations.values():
            conversations.append(self._get_conversation_summary(conv))
        
        # Load recent conversations from disk
        for conv_file in sorted(self.conversations_dir.glob("*.json"), 
                               key=lambda x: x.stat().st_mtime, reverse=True):
            if len(conversations) >= limit:
                break
            
            try:
                conversation_id = conv_file.stem
                if conversation_id not in self.active_conversations:
                    conv = self._load_conversation(conversation_id)
                    if conv:
                        conversations.append(self._get_conversation_summary(conv))
            except Exception:
                continue
        
        return conversations[:limit]
    
    def cleanup_old_conversations(self, days: int = 7):
        import time
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        for conv_file in self.conversations_dir.glob("*.json"):
            if conv_file.stat().st_mtime < cutoff_time:
                try:
                    conversation_id = conv_file.stem
                    if conversation_id in self.active_conversations:
                        del self.active_conversations[conversation_id]
                    conv_file.unlink()
                except Exception:
                    continue
    
    def _save_conversation(self, conversation: Dict):
        conversation_file = self.conversations_dir / f"{conversation['id']}.json"
        try:
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save conversation {conversation['id']}: {e}")
    
    def _load_conversation(self, conversation_id: str) -> Optional[Dict]:
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        if not conversation_file.exists():
            return None
        
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
                self.active_conversations[conversation_id] = conversation
                return conversation
        except Exception as e:
            print(f"Failed to load conversation {conversation_id}: {e}")
            return None
    
    def _get_conversation_summary(self, conversation: Dict) -> Dict:
        messages = conversation.get("messages", [])
        last_message = messages[-1] if messages else None
        
        return {
            "id": conversation["id"],
            "created_at": conversation["created_at"],
            "last_activity": conversation["metadata"]["last_activity"],
            "total_questions": conversation["metadata"]["total_questions"],
            "document_ids": conversation.get("document_ids", []),
            "last_question": last_message["question"][:100] + "..." if last_message and len(last_message["question"]) > 100 else last_message["question"] if last_message else None,
            "last_answer_preview": last_message["answer"][:150] + "..." if last_message and len(last_message["answer"]) > 150 else last_message["answer"] if last_message else None
        }

# Global instance
conversation_manager = ConversationManager()