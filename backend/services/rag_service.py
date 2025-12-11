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

def create_default_knowledge_base():
    """Create comprehensive knowledge base for business analysis"""
    knowledge_content = """
    Business Analysis Methodology Guide
    
    Requirements Gathering Best Practices:
    - Start with open-ended questions to understand the big picture
    - Ask "what" before "how" - understand the problem before solutions
    - Identify all stakeholders and their needs
    - Document both functional and non-functional requirements
    - Validate requirements with stakeholders regularly
    
    Effective Questioning Techniques:
    - Use the 5 Whys technique to get to root causes
    - Ask about exceptions and edge cases
    - Inquire about success metrics and KPIs
    - Discuss constraints and limitations early
    - Explore alternative scenarios and "what if" situations
    
    BRD Structure Guide:
    1. Executive Summary - High-level overview
    2. Project Overview - Context and business need
    3. Business Objectives - Measurable goals
    4. Scope - What's included and excluded
    5. Stakeholder Analysis - Who's involved and their interests
    6. Functional Requirements - System capabilities
    7. Non-Functional Requirements - Quality attributes
    8. Constraints and Assumptions - Limitations and presuppositions
    9. Success Criteria - How success will be measured
    10. Timeline and Milestones - Project schedule
    
    Common Business Analysis Frameworks:
    - SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)
    - PESTLE Analysis (Political, Economic, Social, Technological, Legal, Environmental)
    - MOST Analysis (Mission, Objectives, Strategies, Tactics)
    - Business Model Canvas
    - Value Proposition Canvas
    """
    
    # Create knowledge base directory
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    
    # Write comprehensive guide
    with open(os.path.join(KNOWLEDGE_DIR, "ba_comprehensive_guide.txt"), "w") as f:
        f.write(knowledge_content)
    
    print(f"✓ Created comprehensive knowledge base at {KNOWLEDGE_DIR}")

def index_knowledge_base():
    """Index knowledge base files with improved chunking"""
    if not os.path.exists(KNOWLEDGE_DIR):
        print(f"Creating knowledge base directory: {KNOWLEDGE_DIR}")
        create_default_knowledge_base()
    
    documents = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        if fname.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(KNOWLEDGE_DIR, fname))
                documents.extend(loader.load())
                print(f"  Loaded: {fname}")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    if not documents:
        print("No documents to index")
        return
    
    # Improved text splitting
    splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separator="\n",
        length_function=len
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks")
    
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
        
        retriever = vectordb.as_retriever(
            search_kwargs={"k": 2}
        )
        
        print("✓ Retriever loaded successfully")
        return retriever
    except Exception as e:
        print(f"Failed to load retriever: {e}")
        return None

if __name__ == "__main__":
    index_knowledge_base()