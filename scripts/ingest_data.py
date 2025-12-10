import os
import sys
import logging
from pathlib import Path

# Fix MacOS OpenMP Crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "raw" / "insurance_terms.pdf"
VECTOR_DB_DIR = BASE_DIR / "data" / "processed"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Ingest")

def run_ingestion():
    logger.info("🚀 Starting Data Ingestion...")

    # 1. Check Source
    if not PDF_PATH.exists():
        logger.error(f"❌ Source PDF not found: {PDF_PATH}")
        sys.exit(1)

    # 2. Load PDF
    logger.info(f"📄 Loading PDF: {PDF_PATH.name}")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()
    logger.info(f"   Loaded {len(documents)} pages.")

    # 3. Split Text
    logger.info("✂️  Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    logger.info(f"   Created {len(docs)} chunks.")

    # 4. Embed & Save
    logger.info("🧠 Generating Embeddings (all-MiniLM-L6-v2)...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(VECTOR_DB_DIR), index_name="vector_index")
        
        logger.info(f"✅ Success! Index saved to: {VECTOR_DB_DIR}")
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")

if __name__ == "__main__":
    run_ingestion()