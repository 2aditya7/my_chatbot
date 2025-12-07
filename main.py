# main.py
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from typing import List
import asyncio

# Import services
from services import rag_service, ollama_service
from services.ollama_service import sanitize_history  # sanitizer with noise filtering

# Import DB functions
from db.database import create_new_session, get_session_history, save_new_turn
from db.models import create_db_and_tables

# --- 1. Pydantic Input/Output Models ---
class ChatInput(BaseModel):
    session_id: str
    user_message: str

class SessionResponse(BaseModel):
    session_id: str

# Global retriever reference (initialized on startup)
RAG_RETRIEVER = None

app = FastAPI(
    title="Conversational BA Agent",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "Welcome to the Conversational BA Chatbot API 🚀"}

# Startup event: create DB tables and initialize RAG
@app.on_event("startup")
def on_startup():
    global RAG_RETRIEVER
    create_db_and_tables()
    try:
        RAG_RETRIEVER = rag_service.get_retriever()
        if RAG_RETRIEVER is None:
            print("RAG retriever initialization returned None.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize RAG Retriever: {e}")
        RAG_RETRIEVER = None

# --- Endpoints ---
@app.post("/session/new", response_model=SessionResponse)
def start_new_chat():
    session_id = create_new_session()
    return SessionResponse(session_id=session_id)


@app.post("/chat")
async def chat_endpoint(input_data: ChatInput):

    session_id = input_data.session_id
    user_message = input_data.user_message

    if RAG_RETRIEVER is None:
        raise HTTPException(status_code=503, detail="RAG service unavailable.")

    # Load history from DB
    history = get_session_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found. Create a new one.")

    # CLEAN CORRUPTED / IRRELEVANT HISTORY
    clean_history = sanitize_history(history)

    print("\n=== CLEANED HISTORY SENT TO MODEL ===")
    for entry in clean_history:
        print(f"{entry.get('role')}: {entry.get('content')[:150]}")
    print("=====================================\n")

    # STREAM RESPONSE
    async def response_generator():
        collected = []

        stream = ollama_service.generate_ba_chat_stream(
            clean_history,
            user_message,
            RAG_RETRIEVER
        )

        try:
            async for chunk in stream:
                if not isinstance(chunk, str):
                    chunk = str(chunk)
                yield chunk
                collected.append(chunk)

        except Exception as e:
            print(f"Streaming error: {e}")
            yield "\n[ERROR: Failed to complete response.]"
            return

        final_response = "".join(collected).strip()
        if final_response:
            try:
                save_new_turn(session_id, user_message, final_response)
            except Exception as e:
                print(f"DB save error: {e}")

    return StreamingResponse(response_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    print("Running Conversational BA Agent on http://localhost:8000")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
