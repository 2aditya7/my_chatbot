# main.py (FINAL VERSION WITH RAG INTEGRATION)
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv() 

import services.ollama_service as ollama_service
import services.gemini_service as gemini_service 
import services.rag_service as rag_service 

# --- RAG Component Initialization ---
RAG_RETRIEVER = rag_service.get_retriever()

if RAG_RETRIEVER is None:
    print("WARNING: RAG retriever failed to load. The chatbot will only use general LLM knowledge.")
else:
    print("RAG Retriever initialized successfully.")

ollama_streamer = ollama_service.generate_ba_analysis_ollama_stream
gemini_analyser = gemini_service.generate_ba_analysis_gemini 

# --- Helper function for Streaming Fix ---
def streaming_response_generator(generator):
    """Encodes generator output to bytes for StreamingResponse compatibility."""
    for chunk in generator:
        yield chunk.encode("utf-8")

# --- 1. Pydantic Input Model ---
class BAInput(BaseModel):
    business_req: str 

# --- 2. FastAPI App Instance ---
app = FastAPI(title="Hybrid BA Chatbot API (Ollama + Gemini + RAG)")

# --- 3. Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Hybrid BA API is running! Check /docs for analysis endpoints."}


@app.post("/analyze/{llm_service}", tags=["Analysis"])
def analyze_requirement(llm_service: str, input_data: BAInput):
    
    llm_service = llm_service.lower()
    business_req = input_data.business_req

    if llm_service == "ollama":
        # RAG INTEGRATION: Pass the retriever to the Ollama service
        return StreamingResponse(
            streaming_response_generator(ollama_streamer(business_req, retriever=RAG_RETRIEVER)),
            media_type="text/plain" 
        )
    elif llm_service == "gemini":
        # Gemini remains basic for now
        analysis_text = gemini_analyser(business_req)
        return {"model_used": llm_service, "analysis": analysis_text}
    else:
        raise HTTPException(
            status_code=400, 
            detail="Invalid LLM service. Must be 'ollama' or 'gemini'."
        )