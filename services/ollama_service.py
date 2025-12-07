# services/ollama_service.py
import os
import asyncio
from dotenv import load_dotenv
from ollama import Client
from typing import List, Dict, AsyncGenerator

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")

# ---- OLLAMA CLIENT ----
try:
    ollama_client = Client(host=OLLAMA_API_URL)
    print(f"Ollama client initialized at: {OLLAMA_API_URL}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to Ollama at {OLLAMA_API_URL}: {e}")
    ollama_client = None

# ---- CLEAN HISTORY SANITIZER ----
def sanitize_history(history: List[Dict[str, str]], max_turns: int = 10) -> List[Dict[str, str]]:
    """
    Keep only recent, relevant conversation turns.
    Limit to last N user/assistant pairs to prevent context pollution.
    """
    cleaned = []
    IRRELEVANT_KEYWORDS = [
        "recipe", "make tea", "how to make", "puzzle", "riddle", "joke", "game"
    ]

    for msg in history:
        # Skip system messages from history
        if msg.get("role") == "system":
            continue
        
        content = (msg.get("content") or "").lower()
        
        # Skip obviously irrelevant messages
        if any(k in content for k in IRRELEVANT_KEYWORDS):
            continue
            
        cleaned.append(msg)
    
    # Keep only recent history to prevent context overload
    return cleaned[-max_turns*2:] if len(cleaned) > max_turns*2 else cleaned

# ---- IMPROVED SYSTEM PROMPTS ----
BA_GATHERING_PROMPT = """You are a Senior Business Analyst conducting a requirements gathering session.

STRICT RULES - FOLLOW THESE EXACTLY:
1. Your ONLY job right now is to ask questions to understand the project
2. Ask 1-2 clear, specific questions per response
3. Base your questions on what the user has already told you
4. DO NOT create any documents, specifications, or requirements yet
5. DO NOT propose solutions or make recommendations
6. DO NOT answer business scenarios - ask questions about them instead

WHAT TO ASK ABOUT:
- Project goals and objectives
- Target users and stakeholders  
- Key features and functionality
- Business constraints and limitations
- Success criteria and metrics
- Timeline and budget (if relevant)
- Risks and concerns

STOP GATHERING when user says:
- "Generate requirements"
- "Create BRD"
- "I'm done answering questions"
- "Create documentation"

{context}

Remember: You are gathering information, not providing solutions yet."""

BA_GENERATION_PROMPT = """You are a Senior Business Analyst creating formal documentation.

The user has finished answering questions. Now create:
- Business Requirements Document (BRD)
- Functional Requirements
- Non-Functional Requirements  
- Use Cases or User Stories
- Acceptance Criteria

Use ONLY the information from this conversation. DO NOT add details not discussed.

{context}"""

GENERAL_PROMPT = """You are a helpful AI assistant. Answer questions clearly and concisely."""

# ---- MODE DETECTION ----
def detect_mode(user_message: str, history: List[Dict]) -> str:
    """
    Returns: 'ba_generate', 'ba_gather', or 'general'
    """
    msg = user_message.lower()
    
    # Check if user wants to generate docs
    GENERATE_TRIGGERS = [
        "generate requirements", "create brd", "create documentation",
        "make the document", "generate document", "i'm done", "create requirements"
    ]
    
    if any(trigger in msg for trigger in GENERATE_TRIGGERS):
        return "ba_generate"
    
    # Check if this is BA-related conversation
    BA_KEYWORDS = [
        "business analyst", "requirements", "functional", "non-functional",
        "nfr", "brd", "use case", "user story", "acceptance criteria",
        "project", "stakeholder", "feature"
    ]
    
    # Check current message or recent history
    recent_messages = " ".join([h.get("content", "") for h in history[-4:]])
    combined_context = (msg + " " + recent_messages).lower()
    
    if any(keyword in combined_context for keyword in BA_KEYWORDS):
        return "ba_gather"
    
    return "general"

# ---- IMPROVED RAG RETRIEVAL ----
def get_relevant_context(user_message: str, retriever, min_relevance: float = 0.5) -> str:
    """
    Get RAG context only if it's actually relevant.
    """
    if not retriever:
        return ""
    
    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(user_message)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(user_message)
        else:
            return ""

        if not docs:
            return ""
        
        # Filter and format relevant docs
        context_parts = []
        for doc in docs[:3]:  # Limit to top 3 most relevant
            content = getattr(doc, "page_content", str(doc))
            if content and len(content) > 50:  # Skip very short snippets
                context_parts.append(content)
        
        if context_parts:
            return "\n\n--- REFERENCE DOCUMENTS (use only if relevant) ---\n" + "\n\n".join(context_parts)
        
        return ""
    
    except Exception as e:
        print(f"RAG error: {e}")
        return ""

# ---- MAIN STREAMING METHOD ----
async def generate_ba_chat_stream(
    session_history: List[Dict[str, str]],
    latest_user_message: str,
    retriever
) -> AsyncGenerator[str, None]:

    if ollama_client is None:
        yield "Ollama service unavailable. Check Ollama server."
        return

    # Detect conversation mode
    mode = detect_mode(latest_user_message, session_history)
    
    # Get RAG context only for BA modes
    context = ""
    if mode.startswith("ba"):
        context = get_relevant_context(latest_user_message, retriever)
    
    # Select appropriate system prompt
    if mode == "ba_gather":
        system_prompt = BA_GATHERING_PROMPT.format(context=context)
    elif mode == "ba_generate":
        system_prompt = BA_GENERATION_PROMPT.format(context=context)
    else:
        system_prompt = GENERAL_PROMPT
    
    # Build message list with cleaned history
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    cleaned_history = sanitize_history(session_history, max_turns=8)
    messages.extend(cleaned_history)
    messages.append({"role": "user", "content": latest_user_message})

    # Stream from Ollama with proper parameters
    try:
        response_stream = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.1,  # Lower = more focused and obedient
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048,  # Max tokens
            }
        )

        for chunk in response_stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            await asyncio.sleep(0.001)

    except Exception as e:
        error_text = f"Error during LLM generation: {e}"
        print(error_text)
        yield error_text