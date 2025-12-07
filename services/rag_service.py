# services/rag_service.py
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Modern LangChain-like imports (adapt to your installed libs)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, '..', 'knowledge_base')
COLLECTION_NAME = "ba_knowledge_base"
PERSIST_DIR = os.path.join(BASE_DIR, '..', 'chroma_db')

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY missing in .env")
    sys.exit(1)

try:
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_KEY
    )
except Exception as e:
    print("Failed to initialize embeddings:", e)
    sys.exit(1)

def index_knowledge_base():
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"Knowledge base not found at {KNOWLEDGE_DIR}")
        return

    documents = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(KNOWLEDGE_DIR, fname))
            documents.extend(loader.load())

    if not documents:
        print("No .txt docs found to index.")
        return

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        chunks,
        EMBEDDINGS,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    print("Indexing complete.")

def get_retriever():
    if not os.path.exists(PERSIST_DIR):
        print("Vector store not found; running indexing...")
        index_knowledge_base()

    try:
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=EMBEDDINGS,
            persist_directory=PERSIST_DIR
        )
        return vectordb.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print("Error loading Chroma retriever:", e)
        return None

if __name__ == "__main__":
    index_knowledge_base()
