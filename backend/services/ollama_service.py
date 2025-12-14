import os
import asyncio
from dotenv import load_dotenv
from ollama import AsyncClient
from typing import List, Dict, AsyncGenerator
import re

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:instruct")

try:
    client = AsyncClient(host=OLLAMA_API_URL)
    print(f"✓ Ollama client initialized with model: {OLLAMA_MODEL}")
except Exception as e:
    print(f"✗ Ollama client error: {e}")
    client = None

class OllamaChatService:
    def __init__(self, retriever=None):
        self.retriever = retriever
        self.is_brd_mode = False
        
    GATHERING_PROMPT = """You are a Senior Business Analyst conducting a requirements gathering session.

CRITICAL RULES:
1. Ask ONLY ONE complete question per response
2. Questions must end naturally (no mid-sentence cutting)
3. Wait for user's answer before asking next question
4. After gathering enough information (8-10 exchanges), suggest creating a BRD
5. Also, if user asks you for some suggestions about features of the business, you have to respond by giving the user some relevant suggestions related to features that can be added to the business.

CONVERSATION STRATEGY:
Phase 1: Ask about business idea, problem, value proposition
Phase 2: Ask about users, customers, market
Phase 3: Ask about features, operations, constraints
Phase 4: Ask about budget, timeline, success metrics

FORMAT:
- Single complete question ending with "?"
- No numbered lists or bullet points
- Professional but conversational tone

EXAMPLES OF GOOD QUESTIONS:
- "What specific problem does your business solve?"
- "Who are your primary target customers?"
- "What is your estimated timeline for launch?"

When you have gathered sufficient information, say:
"Based on our discussion, I have enough information to create a comprehensive Business Requirements Document. Would you like me to generate the BRD now?"

Now, based on the conversation history below, ask ONE appropriate question."""

    BRD_TRANSITION_PROMPT = """You are a Senior Business Analyst. The user has requested to generate a Business Requirements Document (BRD).

Respond by:
1. Confirming you'll generate the BRD
2. Mentioning it will be comprehensive and based on our discussion
3. Letting them know it might take a moment

Example response:
"I'll now generate a comprehensive Business Requirements Document based on our discussion. This document will include all requirements we've gathered. Please wait a moment while I create it."

Now respond to the user's request to generate a BRD."""

    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context from knowledge base"""
        if not self.retriever:
            return ""
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            if docs:
                content = docs[0].page_content
                content = re.sub(r'\d+\.\s*', '', content)
                return f"\n[Business Analysis Tip: {content[:100]}...]"
        except Exception:
            return ""
        
        return ""

    def _clean_question(self, response: str) -> str:
        """Clean and format response to ensure it's a single complete question"""

        response = re.sub(r'\d+\.\s*', '', response)

        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        
        if not sentences:
            return "Could you tell me more about your business idea?"
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if sentence.endswith('?'):

                words = sentence.split()
                if len(words) < 3:  
                    continue
                return sentence
            
            question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'can', 'could', 'would', 'will']
            if any(word in sentence.lower().split()[0] for word in question_words):
                if not sentence.endswith('?'):
                    sentence = sentence.rstrip('.!') + '?'
                return sentence
        
        first_sentence = sentences[0].strip()
        if len(first_sentence.split()) > 5:
            if not first_sentence.endswith('?'):
                first_sentence = first_sentence.rstrip('.!') + '?'
            return first_sentence
        
        return "Could you elaborate on that?"

    async def generate_chat_stream(self, session_history: List[Dict], 
                                   user_message: str, retriever=None) -> AsyncGenerator[str, None]:
        """Generate streaming chat response"""
        
        if client is None:
            yield "Error: Ollama is not running. Please start Ollama with 'ollama serve'"
            return
        
        user_msg_lower = user_message.lower()
        wants_brd = any(phrase in user_msg_lower for phrase in 
                       ['generate brd', 'create brd', 'make requirements', 'draft document', 'create document'])
        
        if wants_brd:
            self.is_brd_mode = True
            messages = [{"role": "system", "content": self.BRD_TRANSITION_PROMPT}]
            messages.append({"role": "user", "content": user_message})
        else:
            # Build gathering prompt
            full_prompt = self.GATHERING_PROMPT
            
            # Add conversation history
            if session_history:
                recent = []
                for msg in session_history[-4:]:  # Last 4 exchanges
                    role = "User" if msg["role"] == "user" else "Analyst"
                    recent.append(f"{role}: {msg['content']}")
                
                if recent:
                    full_prompt += f"\n\nRecent conversation:\n" + "\n".join(recent)
            
            # Add RAG context
            if self.retriever:
                rag_context = self._get_relevant_context(user_message)
                if rag_context:
                    full_prompt += f"\n\n{rag_context}"
            
            messages = [{"role": "system", "content": full_prompt}]
            messages.append({"role": "user", "content": user_message})
        
        try:
            # Generate response
            response = await client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options={
                    "temperature": 0.4 if not wants_brd else 0.2,
                    "num_predict": 150 if not wants_brd else 100,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "stop": ["\n\n", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]
                }
            )
            
            full_response = response['message']['content']
            
            # Clean response if not in BRD mode
            if not wants_brd:
                full_response = self._clean_question(full_response)
            
            # Stream response
            for char in full_response:
                yield char
                await asyncio.sleep(0.01)
            
        except Exception as e:
            yield f"Error: {str(e)}"