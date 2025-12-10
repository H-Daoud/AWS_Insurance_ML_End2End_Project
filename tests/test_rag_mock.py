from src.rag.engine import SmartRAGEngine

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


def test_pipeline():
    engine = MockSmartRAGEngine()
    query = "How can I file a car insurance claim?"
    answer = engine.generate_response(query)
    print(answer)


if __name__ == "__main__":
    test_pipeline()

