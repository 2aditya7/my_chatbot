# services/rag_service.py (DO NOT CHANGE - Final RAG Indexer)
import os
import sys
from dotenv import load_dotenv

load_dotenv() 

# --- CORRECTED IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma 

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, '..', 'knowledge_base') 
COLLECTION_NAME = "ba_knowledge_base"

# --- 2. EMBEDDING MODEL INITIALIZATION ---
EMBEDDINGS = None 
GEMINI_KEY = os.getenv("GEMINI_API_KEY") 

if not GEMINI_KEY:
    print("\n" + "="*50)
    print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
    print("Please ensure your .env file has GEMINI_API_KEY=\"YOUR_KEY\"")
    print("="*50 + "\n")
    sys.exit(1)

try:
    # Fix: Pass the key explicitly to the constructor
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_KEY 
    )
except Exception as e:
    print("\n" + "="*50)
    print("CRITICAL ERROR: Failed to initialize Gemini Embeddings.")
    print(f"Original Error: {e}")
    print("="*50 + "\n")
    sys.exit(1)


# --- 3. INDEXING FUNCTION ---
def index_knowledge_base():
    """Loads documents, splits them into chunks, and stores them in ChromaDB."""
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"Error: Knowledge base directory not found at {KNOWLEDGE_DIR}")
        return
        
    documents = []
    for file in os.listdir(KNOWLEDGE_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(KNOWLEDGE_DIR, file))
            documents.extend(loader.load())

    if not documents:
        print("No documents found to index.")
        return

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print(f"Indexing {len(docs)} document chunks into ChromaDB...")
    Chroma.from_documents(
        docs, 
        EMBEDDINGS, 
        collection_name=COLLECTION_NAME,
        persist_directory="./chroma_db" 
    ).persist()

    print("Indexing complete. Vector store saved.")

# --- 4. RETRIEVAL FUNCTION (The R in RAG) ---
def get_retriever():
    """Loads the persistent vector store and returns a retriever."""
    if not os.path.exists("./chroma_db"):
        print("CRITICAL: Vector store not found. Running initial indexing...")
        index_knowledge_base()
        
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=EMBEDDINGS,
            persist_directory="./chroma_db"
        )
        
        return vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        print(f"Error loading ChromaDB retriever: {e}")
        return None

# --- 5. INITIAL RUN ---
if __name__ == "__main__":
    index_knowledge_base()