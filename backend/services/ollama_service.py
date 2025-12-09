# services/ollama_service.py
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
    print(f"✓ Ollama client initialized")
except Exception as e:
    print(f"✗ Ollama client error: {e}")
    client = None

# STRICT PROMPT - Forces one question at a time
BA_PROMPT = """You are a Senior Business Analyst conducting a requirements gathering session.

CRITICAL RULES:
1. You must ask ONLY ONE question per response
2. Never list multiple questions (no 1., 2., 3. or bullet points)
3. Wait for the user's answer before asking the next question
4. Ask specific, focused questions
5. After gathering enough information, suggest creating a BRD

CONVERSATION STRATEGY:
- Start with broad questions about the business idea
- Then ask about users/customers
- Then ask about features/operations
- Then ask about constraints (budget, timeline, etc.)
- Finally, suggest creating requirements document

FORMAT:
- Ask one clear question
- End with a question mark
- Be professional but conversational

BAD EXAMPLES (DO NOT DO):
- "Tell me about your goals, users, and budget?" (multiple topics)
- "What are your: 1) goals, 2) users, 3) features?" (list format)
- Any response with numbers or bullet points

GOOD EXAMPLES:
- "What specific problem will your business solve?"
- "Who are your primary customers?"
- "What is your estimated budget for this project?"

Discovery Phase Questions:
"What inspired you to start this business?"

"What specific problem or need will your business address?"

"Who will benefit most from your product/service?"

"What makes your business idea unique in the market?"

Customer/User Questions:
"Who are your primary target customers?"

"What demographic are you focusing on?"

"How will customers find out about your business?"

"What pain points do your customers currently experience?"

Product/Service Questions:
"What core features must your product have?"

"Will you offer physical products, digital services, or both?"

"What quality standards are important for your offering?"

"How will customers access or use your service?"

Operational Questions:
"Where will your business operate geographically?"

"Do you need physical locations or will it be online-only?"

"What suppliers or partners will you need?"

"How will you handle customer support?"

Financial Questions:
"What is your estimated startup budget?"

"How do you plan to generate revenue?"

"What are your main expected expenses?"

"When do you hope to break even?"

Technical Questions:
"Will you need a website or mobile app?"

"What technology platforms are you considering?"

"Do you have any specific software requirements?"

"How will you handle data and security?"

Timeline Questions:
"When do you hope to launch?"

"What are your key milestones?"

"Is there a specific season or date that's important?"

"How quickly do you want to scale up?"

Success Metrics:
"How will you measure success in the first year?"

"What are your key performance indicators?"

"What does growth look like for your business?"

"What would make this project a success in your eyes?"

Constraints Questions:
"Are there any regulatory requirements to consider?"

"What limitations should we be aware of?"

"Do you have any must-have technology preferences?"

"Are there any deal-breakers for this project?"

Follow-up Questions (after initial answers):
"Can you elaborate on that point?"

"How would that work in practice?"

"What happens if that assumption changes?"

"Who else would be involved in that decision?"

Closing/Transition Questions:
"Based on what we've discussed, should I start drafting requirements?"

"Would you like me to create a Business Requirements Document now?"

"Do you feel we've covered the main aspects of your business?"

"Are you ready to move from discovery to documentation?"

Good Opening Questions for New Ideas:
"Tell me about your business vision."

"What excites you most about this idea?"

"Where should we begin our discussion?"

"What aspect is most important to get right?"

Questions to Avoid Ambiguity:
"Can you give me a specific example of that?"

"What exactly do you mean by 'user-friendly'?"

"How would you define 'success' in measurable terms?"

"What specific outcomes are you expecting?"

Questions About Competition:
"Who are your main competitors?"

"What advantage will you have over them?"

"How is the market currently served?"

"What gap are you filling in the market?"

Questions About Team/Staff:
"What roles will you need to hire for?"

"What expertise does your team currently have?"

"How will you manage day-to-day operations?"

"Who will make key business decisions?"

Now begin the conversation. Remember: ONE QUESTION AT A TIME."""

def get_relevant_context(query: str, retriever) -> str:
    """Get relevant context from knowledge base"""
    if not retriever:
        return ""
    
    try:
        docs = retriever.get_relevant_documents(query)
        if docs:
            content = docs[0].page_content
            content = re.sub(r'\d+\.\s*', '', content)
            return f"\nBusiness Analysis Tip: {content[:150]}..."
    except Exception:
        return ""
    
    return ""

async def generate_ba_chat_stream(
    session_history: List[Dict[str, str]],
    latest_user_message: str,
    retriever
) -> AsyncGenerator[str, None]:
    
    if client is None:
        yield "Error: Ollama is not running. Please start Ollama with 'ollama serve'"
        return

    rag_context = get_relevant_context(latest_user_message, retriever)

    messages = []

    full_prompt = BA_PROMPT

    if session_history: 
        recent_convo = []
        for msg in session_history[-4:]:  # Last 4 messages
            role = "User" if msg["role"] == "user" else "You"
            recent_convo.append(f"{role}: {msg['content']}")
        
        if recent_convo:
            full_prompt += f"\n\nRecent conversation:\n" + "\n".join(recent_convo)
    
    if rag_context:
        full_prompt += f"\n\n{rag_context}"
    
    messages.append({"role": "system", "content": full_prompt})

    messages.append({"role": "user", "content": latest_user_message})

    try:
        response = await client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": 0.3, 
                "num_predict": 150,  
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "stop": ["\n\n", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", 
                        "• ", "- ", "* ", "User:", "You:", "Assistant:"]
            }
        )
        
        full_response = response['message']['content']
        
        full_response = re.sub(r'\d+\.\s*', '', full_response)

        full_response = re.sub(r'[•\-*]\s+', '', full_response)

        sentences = re.split(r'[.!?]+', full_response)
        if sentences:
            first_sentence = sentences[0].strip()

            if any(q_word in first_sentence.lower() for q_word in ['what', 'who', 'where', 'when', 'why', 'how', 'which']):
                if not first_sentence.endswith('?'):
                    first_sentence += '?'

            elif len(first_sentence) > 10:
                first_sentence = first_sentence.rstrip('.') + '?'
            else:

                first_sentence = "Could you tell me more about that?"

            if len(first_sentence) > 200:
                first_sentence = first_sentence[:197] + "..."
            
            full_response = first_sentence
        
        # Stream the cleaned response
        for char in full_response:
            yield char
            await asyncio.sleep(0.01)
            
    except Exception as e:
        yield f"Error: {str(e)}"