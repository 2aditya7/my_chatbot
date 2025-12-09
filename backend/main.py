from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import uuid
from datetime import datetime
import sqlite3
from services import rag_service
from services.ollama_service import generate_ba_chat_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = sqlite3.connect('chat_sessions.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        messages TEXT,
        created_at TIMESTAMP
    )
''')
conn.commit()

class ChatRequest(BaseModel):
    session_id: str
    message: str

retriever = None

@app.on_event("startup")
async def startup():
    global retriever
    print("🚀 Starting Business Analyst Chatbot")
    
    try:
        retriever = rag_service.get_retriever()
        if retriever:
            print("✓ RAG initialized")
    except Exception as e:
        print(f"⚠ RAG error: {e}")
        retriever = None
    
    print("✅ Server ready at http://localhost:8000")

@app.get("/")
async def home():
    """Serve the chat UI"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Business Analyst Chatbot</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            
            .header {
                background: #2563eb;
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 24px;
                margin-bottom: 5px;
            }
            
            .header p {
                opacity: 0.9;
                font-size: 14px;
            }
            
            .session-info {
                background: #f8fafc;
                padding: 10px 20px;
                border-bottom: 1px solid #e2e8f0;
                font-size: 14px;
                color: #64748b;
            }
            
            #sessionId {
                font-family: monospace;
                background: #e2e8f0;
                padding: 2px 8px;
                border-radius: 4px;
            }
            
            .chat-container {
                height: 500px;
                overflow-y: auto;
                padding: 20px;
            }
            
            .message {
                margin-bottom: 15px;
                display: flex;
                gap: 10px;
            }
            
            .user-message {
                flex-direction: row-reverse;
            }
            
            .avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #3b82f6;
                color: white;
                font-size: 16px;
                flex-shrink: 0;
            }
            
            .user-message .avatar {
                background: #10b981;
            }
            
            .message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                background: #f1f5f9;
                color: #334155;
            }
            
            .user-message .message-content {
                background: #3b82f6;
                color: white;
            }
            
            .message-text {
                line-height: 1.5;
            }
            
            .message-time {
                font-size: 11px;
                color: #94a3b8;
                margin-top: 5px;
                text-align: right;
            }
            
            .user-message .message-time {
                color: rgba(255,255,255,0.8);
            }
            
            .input-area {
                padding: 20px;
                border-top: 1px solid #e2e8f0;
                display: flex;
                gap: 10px;
            }
            
            #messageInput {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            #messageInput:focus {
                border-color: #3b82f6;
            }
            
            #sendButton {
                padding: 12px 24px;
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.3s;
            }
            
            #sendButton:hover {
                background: #2563eb;
            }
            
            #sendButton:disabled {
                background: #94a3b8;
                cursor: not-allowed;
            }
            
            .typing-indicator {
                padding: 0 20px 20px;
                display: none;
            }
            
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .typing-dots span {
                width: 8px;
                height: 8px;
                background: #94a3b8;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out;
            }
            
            .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
            .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
            
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
            
            .welcome-message {
                text-align: center;
                padding: 40px 20px;
                color: #64748b;
            }
            
            .welcome-message h2 {
                color: #1e293b;
                margin-bottom: 10px;
            }
            
            .controls {
                padding: 10px 20px;
                background: #f8fafc;
                border-bottom: 1px solid #e2e8f0;
                display: flex;
                gap: 10px;
            }
            
            .controls button {
                padding: 8px 16px;
                background: white;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
            }
            
            .controls button:hover {
                background: #f1f5f9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Business Analyst Assistant</h1>
                <p>Senior Business Analyst - Requirements Gathering & BRD Generation</p>
            </div>
            
            <div class="session-info">
                Session: <span id="sessionId">---</span>
            </div>
            
            <div class="controls">
                <button onclick="newSession()">New Session</button>
                <button onclick="clearChat()">Clear Chat</button>
                <button onclick="exportBRD()" id="exportBtn" style="display:none;">Export BRD</button>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="welcome-message" id="welcomeMessage">
                    <h2>Welcome! 👋</h2>
                    <p>I'm your Senior Business Analyst. I'll help you analyze your business idea and create requirements.</p>
                    <p>Start by describing your business idea below.</p>
                </div>
                <div id="messages"></div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="messageInput" placeholder="Describe your business idea..." autocomplete="off">
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <script>
            let sessionId = null;
            
            // Initialize
            async function init() {
                await createSession();
                setupEventListeners();
            }
            
            // Create new session
            async function createSession() {
                try {
                    const response = await fetch('/session/new', { method: 'POST' });
                    const data = await response.json();
                    sessionId = data.session_id;
                    document.getElementById('sessionId').textContent = sessionId.substring(0, 8) + '...';
                    clearChat();
                    addBotMessage("Hello! I'm your Senior Business Analyst. Please describe your business idea or project.");
                } catch (error) {
                    console.error('Failed to create session:', error);
                    addBotMessage("Error: Failed to create session. Please refresh.");
                }
            }
            
            // Send message
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message || !sessionId) return;
                
                // Add user message
                addUserMessage(message);
                input.value = '';
                input.disabled = true;
                document.getElementById('sendButton').disabled = true;
                
                // Show typing
                document.getElementById('typingIndicator').style.display = 'block';
                document.getElementById('welcomeMessage').style.display = 'none';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            session_id: sessionId, 
                            message: message 
                        })
                    });
                    
                    if (!response.ok) throw new Error('Server error');
                    
                    // Read stream
                    const reader = response.body.getReader();
                    let botMessage = '';
                    const messageDiv = addBotMessage('');
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = new TextDecoder().decode(value);
                        botMessage += chunk;
                        messageDiv.querySelector('.message-text').textContent = botMessage;
                        scrollToBottom();
                    }
                    
                    // Check if BRD was mentioned
                    if (botMessage.toLowerCase().includes('brd') || 
                        botMessage.toLowerCase().includes('requirements document')) {
                        document.getElementById('exportBtn').style.display = 'inline-block';
                    }
                    
                } catch (error) {
                    console.error('Error:', error);
                    addBotMessage('Sorry, there was an error. Please try again.');
                } finally {
                    // Hide typing and re-enable input
                    document.getElementById('typingIndicator').style.display = 'none';
                    input.disabled = false;
                    document.getElementById('sendButton').disabled = false;
                    input.focus();
                }
            }
            
            // UI Helper functions
            function addUserMessage(text) {
                return addMessage('user', text);
            }
            
            function addBotMessage(text) {
                return addMessage('bot', text);
            }
            
            function addMessage(type, text) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}-message`;
                
                const avatar = document.createElement('div');
                avatar.className = 'avatar';
                avatar.textContent = type === 'user' ? '👤' : '🤖';
                
                const content = document.createElement('div');
                content.className = 'message-content';
                
                const textDiv = document.createElement('div');
                textDiv.className = 'message-text';
                textDiv.textContent = text;
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                timeDiv.textContent = new Date().toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                });
                
                content.appendChild(textDiv);
                content.appendChild(timeDiv);
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(content);
                messagesDiv.appendChild(messageDiv);
                
                scrollToBottom();
                return messageDiv;
            }
            
            function clearChat() {
                document.getElementById('messages').innerHTML = '';
                document.getElementById('welcomeMessage').style.display = 'block';
                document.getElementById('exportBtn').style.display = 'none';
                addBotMessage("Chat cleared. Please describe your business idea.");
            }
            
            async function newSession() {
                if (confirm('Start a new session?')) {
                    await createSession();
                }
            }
            
            async function exportBRD() {
                alert('BRD export feature would generate requirements document here.');
                // In future: call API to generate and download BRD
            }
            
            function scrollToBottom() {
                const container = document.getElementById('chatContainer');
                container.scrollTop = container.scrollHeight;
            }
            
            function setupEventListeners() {
                const input = document.getElementById('messageInput');
                const sendBtn = document.getElementById('sendButton');
                
                // Send on Enter
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        sendMessage();
                    }
                });
                
                // Focus input
                input.focus();
            }
            
            // Initialize when page loads
            window.onload = init;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/session/new")
def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    
    cursor.execute(
        "INSERT INTO sessions (session_id, messages, created_at) VALUES (?, ?, ?)",
        (session_id, json.dumps([]), datetime.now().isoformat())
    )
    conn.commit()
    
    return {"session_id": session_id}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests"""
    # Get session from database
    cursor.execute("SELECT messages FROM sessions WHERE session_id = ?", (request.session_id,))
    result = cursor.fetchone()
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Parse messages
    messages = json.loads(result[0])
    
    async def stream_response():
        nonlocal messages
        response_chunks = []
        
        # Generate response using Ollama
        async for chunk in generate_ba_chat_stream(
            messages,
            request.message,
            retriever
        ):
            yield chunk
            response_chunks.append(chunk)
        
        # Save to database
        full_response = "".join(response_chunks)
        
        # Update messages
        messages.append({"role": "user", "content": request.message})
        messages.append({"role": "assistant", "content": full_response})
        
        # Keep last 20 messages
        if len(messages) > 20:
            messages = messages[-20:]
        
        cursor.execute(
            "UPDATE sessions SET messages = ? WHERE session_id = ?",
            (json.dumps(messages), request.session_id)
        )
        conn.commit()
    
    return StreamingResponse(stream_response(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)