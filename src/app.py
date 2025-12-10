import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add 'src' parent to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.rag.engine import SmartRAGEngine
from src.utils.logger import setup_logger

logger = setup_logger("API_Server")

app = FastAPI(title="HUK-COBURG RAG API")

# Initialize Engine
rag_engine = SmartRAGEngine()

# Global Crash Handler
@app.middleware("http")
async def global_crash_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.critical(f"🔥 UNHANDLED API CRASH: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Critical System Error. Check reports/system_errors.log"}
        )

# Data Models
class QueryRequest(BaseModel):
    query: str

class EmbedRequest(BaseModel):
    text: str

# Endpoints
@app.get("/health")
def health_check():
    status = "healthy" if rag_engine.bedrock_runtime else "degraded"
    return {"status": status}

@app.post("/rag/query")
def query_rag(request: QueryRequest):
    logger.info(f"📩 Received query: {request.query}")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    response = rag_engine.generate_response(request.query)
    return {"answer": response}

@app.post("/rag/embed")
def get_embedding(request: EmbedRequest):
    # Utility endpoint for debugging
    if not rag_engine.embeddings:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        vector = rag_engine.embeddings.embed_query(request.text)
        return {"embedding": vector[:5] + ["... (truncated)"]}
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))