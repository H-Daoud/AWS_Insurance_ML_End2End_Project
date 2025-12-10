# FastAPI_app/app.py
import sys
import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from dev.env or prod.env
load_dotenv("configs/dev.env")

# Ensure project src/ is in Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.engine import SmartRAGEngine

# -------------------------------
# Base Path Reference Example
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PDF_PATH = BASE_DIR / "data" / "raw" / "insurance_terms.pdf"

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastAPI_Timer")

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="HUK-COBURG Insurance RAG API")

# Middleware to log request time
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request {request.url.path} completed in {process_time:.3f}s")
    return response

# -------------------------------
# Initialize RAG Engine Singleton
# -------------------------------
try:
    rag_engine = SmartRAGEngine()
    logger.info("✅ SmartRAGEngine initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG Engine: {e}")
    rag_engine = None

# -------------------------------
# Pydantic Models
# -------------------------------
class RAGQuery(BaseModel):
    query: str

class EmbedRequest(BaseModel):
    text: str

# -------------------------------
# FastAPI Endpoints
# -------------------------------
@app.get("/health")
def health():
    if rag_engine is None:
        return {"status": "error", "message": "Engine failed to load"}
    return {"status": "ok", "engine": "running"}

@app.post("/rag/query")
def rag_query(request: RAGQuery):
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is unavailable")
    try:
        answer = rag_engine.generate_response(request.query)
        return {
            "query": request.query,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail="Internal error during LLM generation")

@app.post("/rag/embed")
def rag_embed(request: EmbedRequest):
    if not rag_engine or not rag_engine.embeddings:
        raise HTTPException(status_code=503, detail="Embeddings model unavailable")
    try:
        vector = rag_engine.embeddings.embed_query(request.text)
        return {"embedding": vector}
    except Exception as e:
        logger.error(f"❌ Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Embedding failed")