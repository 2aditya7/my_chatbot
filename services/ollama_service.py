# services/ollama_service.py (FINAL VERSION WITH RAG AND OLLAMA CONNECTION FIX)
import os
from dotenv import load_dotenv
from ollama import Client, generate

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or "phi"

# --- SIMPLIFIED CONNECTION BLOCK (FIXES STARTUP CRASH) ---
if OLLAMA_API_URL:
    ollama_client = Client(host=OLLAMA_API_URL)
    print(f"Ollama client initialized with URL: {OLLAMA_API_URL}")
else:
    print("CRITICAL ERROR: OLLAMA_API_URL is missing. Ollama service will be unavailable.")
    ollama_client = None
# --- END SIMPLIFIED BLOCK ---


# --- RAG Prompt Template ---
RAG_PROMPT_TEMPLATE = """
You are a highly skilled Business Analyst. Your primary task is to decompose the provided BUSINESS REQUIREMENT into a complete set of project components.

CONTEXT (Use this for Non-Functional Requirements):
---
{context}
---

INSTRUCTION:
Based on the BUSINESS REQUIREMENT, generate the following outputs in a clear, structured list format:

1.  **Functional Requirements (FRs):** List 3-5 specific, testable actions the system must perform.
2.  **Non-Functional Requirements (NFRs):** List 2-3 requirements. If the CONTEXT is relevant, use it to generate the NFRs (e.g., performance, security, accessibility). If the context is not relevant, generate standard NFRs.
3.  **User Story:** Create one User Story following the standard format (As a [Role], I want [Goal], so that [Reason/Benefit]).
4.  **User Journey (Key Steps):** Outline 4-5 high-level steps a user takes to complete the core task mentioned in the requirement.

BUSINESS REQUIREMENT: "{question}"

ANALYSIS:
"""

def generate_ba_analysis_ollama_stream(business_req: str, retriever=None):
    if ollama_client is None:
        yield '{"error": "Ollama service is unavailable."}'
        return
        
    # 1. Retrieval (Search the chroma_db)
    context = ""
    if retriever:
        docs = retriever.invoke(business_req) 
        context = "\n---\n".join([doc.page_content for doc in docs])
    
    # 2. Augmentation (Create the final prompt with context)
    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=business_req)

    # 3. Stream the result from Ollama
    try:
        response_stream = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=final_prompt,
            stream=True,
            options={"temperature": 0.3}
            # *** IMPORTANT: DO NOT include format="json" here. Ollama streams text by default. ***
        )
        
        # --- CORRECTED STREAM ITERATION ---
        for chunk in response_stream:
            # Check if the 'response' key exists and yield its value
            if 'response' in chunk:
                yield chunk['response']
        # --- END CORRECTED ITERATION ---

    except Exception as e:
        # If the model fails during generation (e.g., model not loaded), this catches it.
        print(f"Ollama generation failed during streaming: {e}")
        yield f'{"error": "Ollama generation failed: Check if model is loaded."}'

# Placeholder function for the initial simple Gemini analysis
def generate_ba_analysis_gemini(business_req: str):
    # This will be updated later, for now, it's just a placeholder for the gemini endpoint
    return f"Gemini analysis placeholder for: {business_req}"