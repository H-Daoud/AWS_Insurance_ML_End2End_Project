# tests/test_rag.py
import pytest
from fastapi.testclient import TestClient
from src.app import app  # Your FastAPI app
from src.rag.engine import SmartRAGEngine

# -------------------------------
# Mock Engine
# -------------------------------
class MockSmartRAGEngine(SmartRAGEngine):
    def initialize_bedrock(self):
        # Skip real Bedrock connection
        self.bedrock = None
        self.model_id = "mock-model"

    def generate_response(self, query: str) -> str:
        # Get intent & FAISS context
        intent = self.route_intent(query)
        context = "No policy documents found."
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
        # Return a mock response
        return f"[MOCK RESPONSE] Intent: {intent}\nContext:\n{context}\nAnswer: This is a mock response."

# Replace real engine with mock
app.dependency_overrides = {}
mock_engine = MockSmartRAGEngine()
from src.app import rag_engine
rag_engine = mock_engine

# -------------------------------
# TestClient
# -------------------------------
client = TestClient(app)

# -------------------------------
# Tests
# -------------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("Health Endpoint:", data)

def test_rag_query():
    payload = {"query": "What do customers say about claim delays?"}
    response = client.post("/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    print("RAG Query Answer:", data["answer"])

def test_rag_embed():
    payload = {"text": "Sample text to embed."}
    response = client.post("/rag/embed", json=payload)
    # Since embeddings are real, just check structure
    if response.status_code == 503:
        print("Embeddings model unavailable (mock)")
    else:
        data = response.json()
        assert "embedding" in data
        print("RAG Embed:", data["embedding"][:5], "...")  # Print first 5 values
