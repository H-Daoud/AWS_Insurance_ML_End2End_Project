# src/rag/engine.py
import os
import sys
import numpy as np
import onnxruntime as ort
import boto3
import json
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))
from src.utils.logger import setup_logger

logger = setup_logger("SmartRAGEngine")


class SmartRAGEngine:
    def __init__(self):
        logger.info("🚀 Initializing SmartRAGEngine...")
        self.initialize_bedrock()
        self.initialize_router()
        self.initialize_retrieval()

    # -------------------------
    # Bedrock Initialization
    # -------------------------
    def initialize_bedrock(self):
        model_id = os.getenv("BEDROCK_MODEL_ID")
        region = os.getenv("AWS_REGION")
        api_key = os.getenv("BEDROCK_API_KEY")
        api_key_name = os.getenv("BEDROCK_API_KEY_ID")

        if not all([model_id, region, api_key, api_key_name]):
            logger.critical("❌ Missing Bedrock environment variables.")
            self.bedrock = None
            self.model_id = None
            return

        try:
            # Bedrock API key auth via boto3
            self.bedrock = boto3.client(
                "bedrock",
                region_name=region,
                aws_access_key_id=api_key_name,
                aws_secret_access_key=api_key,
            )
            self.model_id = model_id
            logger.info(f"✅ Bedrock client initialized for model: {self.model_id}")
        except Exception as e:
            logger.critical(f"❌ Bedrock initialization failed: {e}", exc_info=True)
            self.bedrock = None
            self.model_id = None

    # -------------------------
    # Intent Router (ONNX)
    # -------------------------
    def initialize_router(self):
        model_path = BASE_DIR / "models" / "huk_distilbert.onnx"
        tokenizer_path = BASE_DIR / "models" / "tokenizer_files"

        self.intent_labels = ["Claim", "Service", "Policy"]
        self.ort_session = None
        self.tokenizer = None

        try:
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            if model_path.exists():
                self.ort_session = ort.InferenceSession(str(model_path))
                logger.info("✅ Intent Router loaded.")
            else:
                logger.warning("⚠️ ONNX file missing. Router disabled.")
        except Exception as e:
            logger.error(f"❌ Router init failed: {e}")

    # -------------------------
    # Retrieval (FAISS)
    # -------------------------
    def initialize_retrieval(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_path = BASE_DIR / "data" / "processed" / "vector_index.faiss"

            if vector_path.exists():
                self.vector_store = FAISS.load_local(
                    str(vector_path.parent),
                    self.embeddings,
                    index_name="vector_index",
                    allow_dangerous_deserialization=True,
                )
                logger.info("✅ FAISS vector store loaded.")
            else:
                logger.error("❌ Missing FAISS index. Run ingest_data.py.")
                self.vector_store = None
        except Exception as e:
            logger.critical(f"🔥 Retrieval init failed: {e}", exc_info=True)
            self.vector_store = None

    # -------------------------
    # Intent Routing
    # -------------------------
    def route_intent(self, query: str) -> str:
        if not self.ort_session:
            return "General"
        try:
            inputs = self.tokenizer(query, return_tensors="np", padding=True, truncation=True)
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
            logits = self.ort_session.run(None, ort_inputs)[0]
            predicted = np.argmax(logits, axis=1)[0]
            return self.intent_labels[predicted] if predicted < len(self.intent_labels) else "General"
        except:
            return "General"

    # -------------------------
    # LLM Generation via Bedrock
    # -------------------------
    def generate_response(self, query: str) -> str:
        if not self.bedrock:
            return "❌ Bedrock unavailable."

        try:
            intent = self.route_intent(query)
            context = "No policy documents found."
            if self.vector_store:
                docs = self.vector_store.similarity_search(query, k=3)
                context = "\n".join([d.page_content for d in docs])

            prompt = f"""
You are a Senior HUK-COBURG Insurance Expert.

User Intent: {intent}

Relevant Policy Context:
{context}

User Question:
{query}

Provide a clear, accurate insurance-expert answer.
"""

            # Bedrock OpenAI-style API payload
            payload = {
                "model": self.model_id,
                "input": prompt,
                "max_tokens": 500,
                "temperature": 0.2
            }

            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload)
            )

            body = json.loads(response["body"].read())
            # Extract the text output (depends on Bedrock model format)
            if "output_text" in body:
                return body["output_text"]
            elif "content" in body and len(body["content"]) > 0:
                return body["content"][0].get("text", "No response from LLM.")
            else:
                return "No response from LLM."

        except Exception as e:
            logger.critical(f"🔥 LLM generation error: {e}", exc_info=True)
            return "Internal LLM generation error."
