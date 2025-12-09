# services/rag_service.py
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, '..', 'knowledge_base')
PERSIST_DIR = os.path.join(BASE_DIR, '..', 'chroma_db')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY missing in .env")
    sys.exit(1)

try:
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    print("✓ Gemini embeddings loaded")
except Exception as e:
    print(f"✗ Embeddings error: {e}")
    sys.exit(1)

def index_knowledge_base():
    """Index knowledge base files"""
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"Creating knowledge base directory: {KNOWLEDGE_DIR}")
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
        
        # Create a simple knowledge base file
        with open(os.path.join(KNOWLEDGE_DIR, "ba_guide.txt"), "w") as f:
            f.write("""
            Business Analysis Basics:
            - Always ask clarifying questions
            - Understand business goals first
            - Identify stakeholders and users
            - Document functional and non-functional requirements
            - Create user stories and acceptance criteria
            """)
    
    documents = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        if fname.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(KNOWLEDGE_DIR, fname))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    if not documents:
        print("No documents to index")
        return
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    try:
        vectordb = Chroma.from_documents(
            chunks,
            EMBEDDINGS,
            persist_directory=PERSIST_DIR
        )
        vectordb.persist()
        print(f"✓ Vector database created at {PERSIST_DIR}")
    except Exception as e:
        print(f"✗ Failed to create vector store: {e}")

def get_retriever():
    """Get a retriever for similarity search"""
    if not os.path.exists(PERSIST_DIR):
        print("Indexing knowledge base...")
        index_knowledge_base()
    
    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=EMBEDDINGS
        )
        return vectordb.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        print(f"Failed to load retriever: {e}")
        return None

if __name__ == "__main__":
    index_knowledge_base()