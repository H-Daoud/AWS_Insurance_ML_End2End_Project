import os
from pathlib import Path
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
TOKENIZER_DIR = MODEL_DIR / "tokenizer_files"

def setup_environment():
    print(f"🔧 Setting up models in: {MODEL_DIR}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download Tokenizer
    print("⬇️  Downloading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokenizer.save_pretrained(TOKENIZER_DIR)
        print(f"✅ Tokenizer saved to {TOKENIZER_DIR}")
    except Exception as e:
        print(f"❌ Tokenizer download failed: {e}")

    # 2. Check/Create ONNX Model
    onnx_path = MODEL_DIR / "huk_distilbert.onnx"
    if onnx_path.exists():
        print(f"✅ ONNX Model found at {onnx_path}")
    else:
        print("⚠️  ONNX Model not found. Creating a PLACEHOLDER model...")
        try:
            # Create a dummy model just so the system doesn't crash
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            dummy_input = tokenizer("claims processing", return_tensors="pt")
            
            torch.onnx.export(
                model, 
                (dummy_input["input_ids"], dummy_input["attention_mask"]), 
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=14
            )
            print(f"✅ Placeholder ONNX model created at {onnx_path}")
        except Exception as e:
            print(f"❌ Failed to create placeholder model: {e}")

    print("\n🎉 Setup Complete. You can now run the system offline.")

if __name__ == "__main__":
    setup_environment()