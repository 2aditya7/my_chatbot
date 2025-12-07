# db/models.py
from sqlmodel import SQLModel, Field, create_engine
from typing import Optional
from datetime import datetime
import os

sqlite_file_name = "chat_history.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=False)

class Message(SQLModel):
    role: str
    content: str

class ChatSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(unique=True, index=True)
    messages_json: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
