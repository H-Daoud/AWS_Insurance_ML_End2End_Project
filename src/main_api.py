from fastapi import FastAPI
from src.utils import get_logger
app = FastAPI()
logger = get_logger("api")
@app.get("/health")
def health():
    return {"status": "healthy"}
