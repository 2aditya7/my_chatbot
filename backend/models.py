# backend/models.py
from sqlmodel import SQLModel, Field, JSON
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

class ChatSession(SQLModel, table=True):
    """Database model for chat sessions"""
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True)
    title: Optional[str] = Field(default="New Business Idea")
    messages: str = Field(default="[]")  # Store as JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    business_domain: Optional[str] = Field(default=None)
    brd_generated: bool = Field(default=False)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Parse messages JSON string to list"""
        return json.loads(self.messages) if self.messages else []
    
    def set_messages(self, messages: List[Dict[str, str]]):
        """Set messages as JSON string"""
        self.messages = json.dumps(messages, ensure_ascii=False)
        self.updated_at = datetime.utcnow()
    
    def add_message(self, role: str, content: str):
        """Add a message to the session"""
        messages = self.get_messages()
        messages.append({"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()})
        
        # Keep only last 20 messages
        if len(messages) > 20:
            messages = messages[-20:]
        
        self.set_messages(messages)
        return messages