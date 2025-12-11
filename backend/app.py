from fastapi import FastAPI, HTTPException, Response
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
from services.ollama_service import OllamaChatService
from services.brd_generator import BRDGenerator
from contextlib import asynccontextmanager

# Database connection - define globally
conn = None
cursor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown"""
    global conn, cursor, retriever, ollama_service, brd_generator
    
    print("🚀 Starting Business Analyst Chatbot")
    
    # Database setup with schema migration
    try:
        # Connect to database
        conn = sqlite3.connect('chat_sessions.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # First, check if old table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions_old'")
        old_table_exists = cursor.fetchone()
        
        if old_table_exists:
            # Drop the old backup if exists
            cursor.execute("DROP TABLE IF EXISTS sessions_old")
        
        # Check current schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'")
        table_info = cursor.fetchone()
        
        if table_info:
            # Table exists, check if it has new columns
            cursor.execute("PRAGMA table_info(sessions)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            print(f"Current table columns: {column_names}")
            
            if 'brd_generated' not in column_names or 'brd_content' not in column_names:
                print("⚠ Updating database schema...")
                
                # Backup old table
                cursor.execute("ALTER TABLE sessions RENAME TO sessions_old")
                
                # Create new table with updated schema
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        messages TEXT,
                        brd_generated BOOLEAN DEFAULT 0,
                        brd_content TEXT,
                        created_at TIMESTAMP
                    )
                ''')
                
                # Copy data from old table
                cursor.execute("""
                    INSERT INTO sessions (session_id, messages, created_at)
                    SELECT session_id, messages, created_at FROM sessions_old
                """)
                
                # Drop old table
                cursor.execute("DROP TABLE sessions_old")
                
                print("✓ Database schema updated successfully")
            else:
                print("✓ Database schema is up to date")
        else:
            # Create table if it doesn't exist
            print("Creating new database table...")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT,
                    brd_generated BOOLEAN DEFAULT 0,
                    brd_content TEXT,
                    created_at TIMESTAMP
                )
            ''')
        
        conn.commit()
        print("✓ Database initialized")
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        # Fallback: create fresh table
        try:
            cursor.execute("DROP TABLE IF EXISTS sessions")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT,
                    brd_generated BOOLEAN DEFAULT 0,
                    brd_content TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()
            print("✓ Created fresh database table")
        except Exception as e2:
            print(f"✗ Critical database error: {e2}")
            raise
    
    # Initialize RAG
    try:
        retriever = rag_service.get_retriever()
        if retriever:
            print("✓ RAG initialized")
    except Exception as e:
        print(f"⚠ RAG error: {e}")
        retriever = None
    
    # Initialize services
    ollama_service = OllamaChatService(retriever)
    brd_generator = BRDGenerator()
    
    print("✅ Server ready at http://localhost:8000")
    
    yield  # App runs here
    
    # Shutdown
    print("🛑 Shutting down...")
    if conn:
        conn.close()
    print("✅ Server shutdown complete")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ExportRequest(BaseModel):
    session_id: str
    format: str = "txt"

# Global variables
retriever = None
ollama_service = None
brd_generator = None

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
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: 95vh;
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
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            #sessionId {
                font-family: monospace;
                background: #e2e8f0;
                padding: 2px 8px;
                border-radius: 4px;
            }
            
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                background: #fafafa;
            }
            
            .message {
                margin-bottom: 15px;
                display: flex;
                gap: 10px;
                animation: fadeIn 0.3s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
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
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .user-message .message-content {
                background: #3b82f6;
                color: white;
            }
            
            .message-text {
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
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
                background: white;
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
                transition: all 0.3s;
            }
            
            #sendButton:hover {
                background: #2563eb;
                transform: translateY(-1px);
            }
            
            #sendButton:disabled {
                background: #94a3b8;
                cursor: not-allowed;
                transform: none;
            }
            
            .typing-indicator {
                padding: 0 20px 20px;
                display: none;
            }
            
            .typing-dots {
                display: flex;
                gap: 4px;
                padding-left: 46px;
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
                animation: fadeIn 0.5s ease-in;
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
                flex-wrap: wrap;
            }
            
            .controls button {
                padding: 8px 16px;
                background: white;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s;
            }
            
            .controls button:hover {
                background: #f1f5f9;
                transform: translateY(-1px);
            }
            
            .mode-indicator {
                display: inline-block;
                padding: 4px 12px;
                background: #dbeafe;
                color: #1e40af;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                margin-left: auto;
            }
            
            .export-options {
                display: none;
                gap: 5px;
                margin-left: 10px;
            }
            
            .export-options button {
                padding: 6px 12px;
                font-size: 12px;
            }
            
            .export-txt { background: #dbeafe; border-color: #93c5fd; }
            .export-doc { background: #fef3c7; border-color: #fcd34d; }
            .export-pdf { background: #fee2e2; border-color: #fca5a5; }
            
            .brd-section {
                background: #f8fafc;
                border-left: 4px solid #3b82f6;
                padding: 15px;
                margin: 15px 0;
                border-radius: 0 8px 8px 0;
            }
            
            .brd-section h4 {
                margin-top: 0;
                color: #1e40af;
            }
            
            .brd-preview {
                max-height: 300px;
                overflow-y: auto;
                background: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #e2e8f0;
                font-family: monospace;
                font-size: 12px;
                white-space: pre-wrap;
            }
            
            .generating-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                display: none;
                justify-content: center;
                align-items: center;
                z-index: 1000;
            }
            
            .generating-modal {
                background: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            
            .progress-bar {
                width: 300px;
                height: 20px;
                background: #e2e8f0;
                border-radius: 10px;
                margin: 20px 0;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background: #3b82f6;
                width: 0%;
                transition: width 0.3s;
                border-radius: 10px;
            }
            
            .error-message {
                background: #fee2e2;
                border-left: 4px solid #dc2626;
                color: #991b1b;
                padding: 10px;
                margin: 10px 0;
                border-radius: 0 4px 4px 0;
            }
        </style>
    </head>
    <body>
        <div class="generating-overlay" id="generatingOverlay">
            <div class="generating-modal">
                <h3>🔧 Generating Business Requirements Document</h3>
                <p>This may take a moment. Please wait...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div id="progressText">Starting...</div>
            </div>
        </div>
        
        <div class="container">
            <div class="header">
                <h1>🤖 Business Analyst Assistant</h1>
                <p>Senior Business Analyst - Requirements Gathering & BRD Generation</p>
            </div>
            
           <div class="session-info">
                <div>
                    <span>Session: <span id="sessionId">---</span></span>
                    <span id="modeIndicator" class="mode-indicator">Gathering Requirements</span>
                </div>
                <div id="exportOptions" class="export-options">
                    <button class="export-txt">TXT</button>
                    <button class="export-doc">DOC</button>
                    <button class="export-pdf">PDF</button>
                </div>
            </div>
            
                <div class="controls">
                    <button id="newSessionBtn">🆕 New Session</button>
                    <button id="clearChatBtn">🗑️ Clear Chat</button>
                    <button id="generateBRDBtn" style="display:none;">📄 Generate BRD</button>
                    <button id="exportToggleBtn" style="display:none;">📥 Export Options</button>
                </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="welcome-message" id="welcomeMessage">
                    <h2>Welcome! 👋</h2>
                    <p>I'm your Senior Business Analyst. I'll help you analyze your business idea and create requirements.</p>
                    <p>Start by describing your business idea below.</p>
                    <div class="brd-section">
                        <h4>💡 How it works:</h4>
                        <p>1. <strong>Discovery Phase</strong>: I'll ask focused questions about your business</p>
                        <p>2. <strong>Requirements Gathering</strong>: We'll discuss features, users, and constraints</p>
                        <p>3. <strong>BRD Generation</strong>: Type "generate BRD" or "create requirements document"</p>
                        <p>4. <strong>Export</strong>: Download as TXT, DOC, or PDF</p>
                    </div>
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
                <input type="text" id="messageInput" placeholder="Describe your business idea or type 'generate BRD'..." autocomplete="off">
                <button id="sendButton">Send</button>
            </div>
        </div>
        
        <script>
            let sessionId = null;
            let isGeneratingBRD = false;
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', function() {
                console.log('DOM loaded, initializing...');
                init();
            });
            
            // Main initialization
            async function init() {
                console.log('Initializing app...');
                await createSession();
                setupEventListeners();
            }
            
            // Create new session
            async function createSession() {
                try {
                    console.log('Creating session...');
                    const response = await fetch('/session/new', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
                    }
                    
                    const data = await response.json();
                    sessionId = data.session_id;
                    console.log('Session ID:', sessionId);
                    
                    // Update UI
                    document.getElementById('sessionId').textContent = sessionId.substring(0, 8) + '...';
                    
                    // Clear any existing chat and show welcome
                    document.getElementById('messages').innerHTML = '';
                    document.getElementById('welcomeMessage').style.display = 'block';
                    addBotMessage("Hello! I'm your Senior Business Analyst. Please describe your business idea or project.");
                    
                } catch (error) {
                    console.error('Failed to create session:', error);
                    alert('Failed to create session. Please refresh the page.');
                }
            }
            
            // Send message function
            async function sendMessage() {
                console.log('sendMessage called');
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                
                if (!message || !sessionId) {
                    console.log('No message or session');
                    return;
                }
                
                if (isGeneratingBRD) {
                    console.log('Already generating BRD');
                    return;
                }
                
                console.log('Processing message:', message);
                
                // Check for BRD generation request
                const brdTriggers = ['generate brd', 'create brd', 'make requirements', 'draft document', 'create document'];
                const shouldGenerateBRD = brdTriggers.some(trigger => message.toLowerCase().includes(trigger));
                
                if (shouldGenerateBRD) {
                    generateBRD();
                    input.value = '';
                    return;
                }
                
                // Normal chat message
                addUserMessage(message);
                input.value = '';
                input.disabled = true;
                document.getElementById('sendButton').disabled = true;
                
                // Show typing indicator
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
                    
                    if (!response.ok) {
                        throw new Error(`Server error: ${response.status}`);
                    }
                    
                    // Stream the response
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
                    
                    // Check if we should show BRD button
                    if (botMessage.toLowerCase().includes('ready for brd') || 
                        botMessage.toLowerCase().includes('enough information') ||
                        botMessage.toLowerCase().includes('generate brd')) {
                        document.getElementById('generateBRDBtn').style.display = 'inline-block';
                    }
                    
                } catch (error) {
                    console.error('Chat error:', error);
                    addBotMessage('Sorry, there was an error. Please try again.');
                } finally {
                    // Reset UI
                    document.getElementById('typingIndicator').style.display = 'none';
                    input.disabled = false;
                    document.getElementById('sendButton').disabled = false;
                    input.focus();
                }
            }
            
            // Generate BRD
            async function generateBRD() {
                console.log('generateBRD called');
                
                if (!sessionId || isGeneratingBRD) {
                    console.log('Cannot generate BRD');
                    return;
                }
                
                isGeneratingBRD = true;
                const input = document.getElementById('messageInput');
                const sendBtn = document.getElementById('sendButton');
                
                // Disable UI
                input.disabled = true;
                sendBtn.disabled = true;
                document.getElementById('generateBRDBtn').disabled = true;
                
                // Show overlay
                document.getElementById('generatingOverlay').style.display = 'flex';
                
                addBotMessage("🔧 Starting BRD generation... This may take a moment.");
                
                try {
                    console.log('Calling generate_brd endpoint');
                    const response = await fetch(`/generate_brd/${sessionId}`, {
                        method: 'POST'
                    });
                    
                    if (!response.ok) {
                        throw new Error(`BRD generation failed: ${response.status}`);
                    }
                    
                    // Stream the BRD
                    const reader = response.body.getReader();
                    let brdContent = '';
                    const messageDiv = addBotMessage('');
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        const chunk = new TextDecoder().decode(value);
                        brdContent += chunk;
                        messageDiv.querySelector('.message-text').textContent = 
                            `🔧 Generating BRD... (${brdContent.length} characters so far)`;
                        scrollToBottom();
                    }
                    
                    console.log('BRD generated, length:', brdContent.length);
                    
                    // Hide overlay
                    document.getElementById('generatingOverlay').style.display = 'none';
                    
                    // Show BRD preview
                    const previewDiv = document.createElement('div');
                    previewDiv.className = 'brd-section';
                    previewDiv.innerHTML = `
                        <h4>📋 Business Requirements Document Generated!</h4>
                        <div class="brd-preview">${brdContent.substring(0, 500)}${brdContent.length > 500 ? '...' : ''}</div>
                        <p><small>Full document (${brdContent.length} characters) ready for export.</small></p>
                    `;
                    
                    const brdMessage = addMessage('bot', '');
                    brdMessage.querySelector('.message-content').appendChild(previewDiv);
                    
                    // Show export options
                    document.getElementById('exportOptions').style.display = 'flex';
                    document.getElementById('exportToggleBtn').style.display = 'inline-block';
                    
                } catch (error) {
                    console.error('BRD generation error:', error);
                    document.getElementById('generatingOverlay').style.display = 'none';
                    addBotMessage(`❌ BRD generation failed: ${error.message}`);
                } finally {
                    // Re-enable UI
                    isGeneratingBRD = false;
                    input.disabled = false;
                    sendBtn.disabled = false;
                    document.getElementById('generateBRDBtn').disabled = false;
                    input.focus();
                }
            }
            
            // Export BRD
            async function handleExport(format) {
                console.log('Exporting as:', format);
                
                if (!sessionId) {
                    console.log('No session ID');
                    return;
                }
                
                const button = event.target;
                const originalText = button.textContent;
                
                try {
                    button.textContent = 'Exporting...';
                    button.disabled = true;
                    
                    const response = await fetch(`/export_brd/${sessionId}?format=${format}`);
                    
                    if (!response.ok) {
                        throw new Error(`Export failed: ${response.status}`);
                    }
                    
                    // Download the file
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    
                    // Get filename
                    let filename = `BRD_${sessionId.substring(0, 8)}.${format}`;
                    const contentDisposition = response.headers.get('content-disposition');
                    if (contentDisposition) {
                        const match = contentDisposition.match(/filename="(.+)"/);
                        if (match) filename = match[1];
                    }
                    
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    
                    addBotMessage(`✅ BRD exported as ${format.toUpperCase()} file.`);
                    
                } catch (error) {
                    console.error('Export error:', error);
                    addBotMessage(`❌ Export failed: ${error.message}`);
                } finally {
                    button.textContent = originalText;
                    button.disabled = false;
                }
            }
            
            // Toggle export options
            function toggleExportOptions() {
                const options = document.getElementById('exportOptions');
                options.style.display = options.style.display === 'flex' ? 'none' : 'flex';
            }
            
            // Clear chat
            function clearChat() {
                console.log('Clearing chat');
                if (confirm('Clear all messages?')) {
                    document.getElementById('messages').innerHTML = '';
                    document.getElementById('welcomeMessage').style.display = 'block';
                    document.getElementById('generateBRDBtn').style.display = 'none';
                    document.getElementById('exportOptions').style.display = 'none';
                    document.getElementById('exportToggleBtn').style.display = 'none';
                    addBotMessage("Chat cleared. Please describe your business idea.");
                }
            }
            
            // Setup all event listeners
            function setupEventListeners() {
                console.log('Setting up event listeners');
                
                // Get all elements
                const sendBtn = document.getElementById('sendButton');
                const newSessionBtn = document.getElementById('newSessionBtn');
                const clearChatBtn = document.getElementById('clearChatBtn');
                const generateBtn = document.getElementById('generateBRDBtn');
                const exportToggleBtn = document.getElementById('exportToggleBtn');
                const input = document.getElementById('messageInput');
                
                // Send button
                sendBtn.addEventListener('click', sendMessage);
                
                // Enter key in input
                input.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        sendMessage();
                    }
                });
                
                // New session button
                newSessionBtn.addEventListener('click', function() {
                    if (confirm('Start a new session? Current conversation will be lost.')) {
                        createSession();
                    }
                });
                
                // Clear chat button
                clearChatBtn.addEventListener('click', clearChat);
                
                // Generate BRD button
                if (generateBtn) {
                    generateBtn.addEventListener('click', generateBRD);
                }
                
                // Export toggle button
                if (exportToggleBtn) {
                    exportToggleBtn.addEventListener('click', toggleExportOptions);
                }
                
                // Export format buttons
                document.querySelectorAll('.export-options button').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const format = this.className.includes('export-txt') ? 'txt' :
                                     this.className.includes('export-doc') ? 'doc' : 'pdf';
                        handleExport(format);
                    });
                });
                
                // Focus input
                input.focus();
                
                console.log('Event listeners set up');
            }
            
            // Helper functions for messages
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
            
            function scrollToBottom() {
                const container = document.getElementById('chatContainer');
                container.scrollTop = container.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/session/new")
def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    
    try:
        cursor.execute(
            "INSERT INTO sessions (session_id, messages, brd_generated, brd_content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, json.dumps([]), 0, "", datetime.now().isoformat())
        )
        conn.commit()
        
        return {"session_id": session_id}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat requests with improved conversation flow"""
    # Get session from database
    try:
        cursor.execute("SELECT messages FROM sessions WHERE session_id = ?", (request.session_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Parse messages
        messages = json.loads(result[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def stream_response():
        nonlocal messages
        response_chunks = []
        
        try:
            # Generate response using Ollama service
            async for chunk in ollama_service.generate_chat_stream(
                messages,
                request.message,
                retriever
            ):
                yield chunk
                response_chunks.append(chunk)
        except Exception as e:
            yield f"Error: {str(e)}"
            return
        
        # Save to database
        full_response = "".join(response_chunks)
        
        # Update messages
        messages.append({"role": "user", "content": request.message})
        messages.append({"role": "assistant", "content": full_response})
        
        # Keep last 30 messages for context
        if len(messages) > 30:
            messages = messages[-30:]
        
        try:
            cursor.execute(
                "UPDATE sessions SET messages = ? WHERE session_id = ?",
                (json.dumps(messages), request.session_id)
            )
            conn.commit()
        except Exception as e:
            print(f"Warning: Failed to save message to database: {e}")
    
    return StreamingResponse(stream_response(), media_type="text/plain")

@app.post("/generate_brd/{session_id}")
async def generate_brd(session_id: str):
    """Generate a Business Requirements Document from session"""
    # Get session messages
    try:
        cursor.execute("SELECT messages FROM sessions WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = json.loads(result[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def stream_brd():
        brd_chunks = []
        
        try:
            # Generate BRD using the generator service
            async for chunk in brd_generator.generate_brd_stream(messages, retriever):
                yield chunk
                brd_chunks.append(chunk)
        except Exception as e:
            yield f"Error generating BRD: {str(e)}"
            return
        
        # Save BRD to database
        full_brd = "".join(brd_chunks)
        try:
            cursor.execute(
                "UPDATE sessions SET brd_generated = 1, brd_content = ? WHERE session_id = ?",
                (full_brd, session_id)
            )
            conn.commit()
        except Exception as e:
            print(f"Warning: Failed to save BRD to database: {e}")
    
    return StreamingResponse(stream_brd(), media_type="text/plain")

@app.get("/export_brd/{session_id}")
async def export_brd(session_id: str, format: str = "txt"):
    """Export BRD in requested format"""
    try:
        cursor.execute("SELECT brd_content FROM sessions WHERE session_id = ? AND brd_generated = 1", (session_id,))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail="BRD not found or not generated")
        
        brd_content = result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"BRD_{session_id[:8]}_{timestamp}"
    
    if format == "txt":
        # Return as plain text
        return Response(
            content=brd_content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}.txt"}
        )
    
    elif format == "doc":
        # Create simple DOC format (RTF)
        rtf_content = create_rtf_document(brd_content)
        return Response(
            content=rtf_content,
            media_type="application/rtf",
            headers={"Content-Disposition": f"attachment; filename={filename}.doc"}
        )
    
    elif format == "pdf":
        try:
            # Try to create PDF if reportlab is installed
            from reportlab.lib.pagesizes import letter
            pdf_content = create_pdf_document(brd_content)
            return Response(
                content=pdf_content,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename={filename}.pdf"}
            )
        except ImportError:
            # Fallback to HTML
            html_content = create_html_document(brd_content)
            return Response(
                content=html_content,
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename={filename}.html"}
            )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

def create_rtf_document(content: str) -> str:
    """Create RTF document from content"""
    # Simple RTF header
    rtf_header = r"""{\rtf1\ansi\deff0 {\fonttbl {\f0 Times New Roman;}}
    \margl1440\margr1440\margt1440\margb1440
    \pard\f0\fs24"""
    
    # Escape RTF special characters
    escaped_content = content.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
    escaped_content = escaped_content.replace('\n', '\\par ')
    
    return rtf_header + escaped_content + "}"

def create_pdf_document(content: str) -> bytes:
    """Create PDF document from content"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib import colors
        import io
        
        # Create buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        # Create styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.HexColor('#1e40af')
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.HexColor('#1e293b')
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        # Split content into paragraphs
        paragraphs = content.split('\n')
        story = []
        
        for para in paragraphs:
            if para.strip():
                if para.startswith('# '):
                    # Main title
                    story.append(Paragraph(para[2:], title_style))
                elif para.startswith('## '):
                    # Section heading
                    story.append(Paragraph(para[3:], heading_style))
                elif para.startswith('### '):
                    # Sub-section
                    story.append(Paragraph(para[4:], styles['Heading3']))
                elif para.startswith('- '):
                    # List item
                    story.append(Paragraph(f"• {para[2:]}", normal_style))
                else:
                    # Normal paragraph
                    story.append(Paragraph(para, normal_style))
                story.append(Spacer(1, 3))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    except ImportError:
        raise ImportError("PDF generation requires reportlab. Please install with: pip install reportlab")

def create_html_document(content: str) -> str:
    """Create HTML document from content (fallback for PDF)"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Business Requirements Document</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #1e40af; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
            h2 {{ color: #1e293b; margin-top: 30px; }}
            h3 {{ color: #374151; margin-top: 20px; }}
            .section {{ margin-bottom: 20px; }}
            .list-item {{ margin-left: 20px; }}
            .metadata {{ color: #6b7280; font-size: 12px; margin-bottom: 30px; }}
            .content {{ white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="metadata">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            Document Type: Business Requirements Document
        </div>
        <div class="content">
            {content.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')}
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)