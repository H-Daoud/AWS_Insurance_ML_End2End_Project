# tests/test_fastapi_mock.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag.engine import SmartRAGEngine

# Mock version of SmartRAGEngine
class MockSmartRAGEngine(SmartRAGEngine):
    def initialize_bedrock(self):
        self.bedrock = None
        self.model_id = "mock-model"

    def generate_response(self, query: str) -> str:
        intent = self.route_intent(query)
        context = "No policy documents found."
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
        return f"[MOCK RESPONSE] Intent: {intent}\nContext:\n{context}\nAnswer: This is a mock response."

# Initialize mock engine
engine = MockSmartRAGEngine()

# FastAPI app
app = FastAPI(title="HUK-COBURG Insurance RAG API (Mock)")

class RAGQuery(BaseModel):
    query: str

class EmbedRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    if not engine:
        return {"status": "error", "message": "Engine failed to load"}
    return {"status": "ok", "engine": "running"}

@app.post("/rag/query")
def rag_query(request: RAGQuery):
    if not engine:
        raise HTTPException(status_code=503, detail="RAG Engine is unavailable")
    try:
        answer = engine.generate_response(request.query)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/embed")
def rag_embed(request: EmbedRequest):
    if not engine or not engine.embeddings:
        raise HTTPException(status_code=503, detail="Embeddings model unavailable")
    try:
        vector = engine.embeddings.embed_query(request.text)
        return {"embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

