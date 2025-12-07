# db/database.py
from sqlmodel import Session, select
from uuid import uuid4
import json
from typing import List, Dict, Optional
from .models import engine, ChatSession

def create_new_session() -> str:
    session_id = str(uuid4())
    new_session = ChatSession(
        session_id=session_id,
        messages_json=json.dumps([])
    )
    with Session(engine) as session:
        session.add(new_session)
        session.commit()
    return session_id

def get_session_history(session_id: str) -> Optional[List[Dict[str, str]]]:
    with Session(engine) as session:
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        result = session.exec(statement).first()
        if result:
            return json.loads(result.messages_json)
        return None

def save_new_turn(session_id: str, user_message: str, assistant_response: str) -> None:
    with Session(engine) as session:
        statement = select(ChatSession).where(ChatSession.session_id == session_id)
        chat_session = session.exec(statement).first()
        if chat_session:
            history_dicts = json.loads(chat_session.messages_json)
            history_dicts.append({"role": "user", "content": user_message})
            history_dicts.append({"role": "assistant", "content": assistant_response})
            chat_session.messages_json = json.dumps(history_dicts)
            session.add(chat_session)
            session.commit()
