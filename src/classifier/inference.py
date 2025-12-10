import sys
import argparse
import logging
from pathlib import Path

# --- 1. Setup Project Path & MacOS Fix ---
# Add the project root to sys.path so we can import src.rag.engine
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# MacOS OpenMP crash fix (standard for this project)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.rag.engine import SmartRAGEngine

# --- 2. Logging Setup ---
# We use a cleaner format for CLI output
logging.basicConfig(level=logging.ERROR)  # Only show errors to keep output clean
logger = logging.getLogger("InferenceCLI")

def run_inference(query: str):
    print(f"\n🤖 Initializing Engine...")
    try:
        engine = SmartRAGEngine()
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return

    print(f"🔍 Analyzing Query: '{query}'")
    print("-" * 50)

    # 1. Check Intent (Optional visibility for debugging)
    intent = engine.route_intent(query)
    print(f"🧭 Detected Intent: {intent}")

    # 2. Get Answer
    print("⏳ Generating Answer (GPT-4o)...")
    response = engine.generate_response(query)
    
    print("-" * 50)
    print("📝 FINAL ANSWER:")
    print(response)
    print("-" * 50)

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="HUK-COBURG RAG Inference CLI")
    parser.add_argument(
        "query", 
        type=str, 
        nargs="?", 
        help="The question you want to ask the bot.",
        default="Is windshield damage covered by my policy?"
    )
    
    args = parser.parse_args()
    
    # Run
    run_inference(args.query)