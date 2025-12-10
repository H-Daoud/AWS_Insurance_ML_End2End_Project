# Project_Doc_6_FASTAPI.md

---

# NOTE: This project now uses AWS as the primary cloud infrastructure. Any references to Azure (e.g., Azure OpenAI, Azure ML, Azure-specific Terraform templates, or Azure deployment scripts) are deprecated and should be ignored for current and future deployments. All instructions, diagrams, and documentation should focus on AWS resources, workflows, and best practices.

## FastAPI Quickstart & Troubleshooting Guide

This document explains how to run, test, and troubleshoot the FastAPI backend for the RAG workflow and sentiment analysis system. It is designed for beginners and new team members.

---

### 1. What is FastAPI?
FastAPI is a modern Python web framework for building APIs. In this project, it powers the backend endpoints for RAG queries and health checks.

---

### 2. How to Run the FastAPI Server

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Start the server**
```bash
python3 -m uvicorn FastAPI_app.app:app --reload --env-file configs/dev.env
```
- The server will run at: [http://localhost:8000](http://localhost:8000)
- You should see `Application startup complete.` in the terminal.

---

### 3. How to Test Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{"status": "ok"}
```

**RAG Query:**
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the process for vehicle insurance claims?"}'
```
Expected response:
```json
{"results": [{"chunk": "Sample result for query: What is the process for vehicle insurance claims?", "score": 1.0}]}
```

**Interactive API Docs:**
- Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for Swagger UI.
- You can test endpoints directly from this page.

---

### 4. Common Errors & Troubleshooting

**Error: `bash: uvicorn: command not found`**
- Solution: Run with Python module syntax:
  ```bash
  python3 -m uvicorn FastAPI_app.app:app --reload --env-file configs/dev.env
  ```

**Error: `curl: (7) Failed to connect to localhost port 8000`**
- Solution: Make sure the FastAPI server is running and you see `Application startup complete.`
- Check for errors in the server terminal.

**Error: `404 Not Found` for / or /favicon.ico**
- Solution: This is normal. Only `/health` and `/rag/query` endpoints are defined.

**Error: No response from /rag/query**
- Solution: Ensure you are sending a POST request with JSON body. Check server logs for errors.
- Try testing with Swagger UI at `/docs`.

**Error: ModuleNotFound or ImportError**
- Solution: Make sure you installed all dependencies with `pip install -r requirements.txt`.

---

### 5. Useful Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Swagger UI for your API](http://localhost:8000/docs)

---

### 6. Tips for Beginners
- Always check the terminal running FastAPI for error messages.
- Use `curl` or Postman for quick endpoint testing.
- Use `/docs` for interactive API exploration.
- If you change code, restart the server to apply changes.
- If you get stuck, check this doc for solutions before asking for help!

---

### 7. RAG Engine Integration & Real Results (Updated)

The `/rag/query` endpoint now uses real translation, embedding, and FAISS vector search logic:
- Translates your question to English if needed
- Embeds the question using DistilBERT
- Searches the FAISS index for the most relevant document chunks
- Returns those chunks and their scores

**Example:**
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Wie melde ich einen Schaden?"}'
```
**Expected response:**
```json
{
  "results": [
    {"chunk": "The insurance claim process involves...", "score": 0.92},
    {"chunk": "Claims must be submitted within...", "score": 0.87}
  ]
}
```

---

### 8. Advanced RAG Endpoints

**Translation Endpoint:**
```bash
curl -X POST http://localhost:8000/rag/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Schadenmeldung", "source_lang": "de", "target_lang": "en"}'
```
Expected response:
```json
{"translated": "Claim notification"}
```

**Embedding Endpoint:**
```bash
curl -X POST http://localhost:8000/rag/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Claim notification"}'
```
Expected response:
```json
{"embedding": [0.123, 0.456, ...]}
```

---

Update this guide as your API evolves or new endpoints are added.
