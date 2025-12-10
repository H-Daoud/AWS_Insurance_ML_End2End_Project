# aws/score.py
# AWS entry script for ONNX inference
# Place model and tokenizer in the Docker image or S3 bucket

import os
import json
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

# -------------------------------
# Global variables for session, tokenizer, labels
# -------------------------------
session = None
tokenizer = None
label_map = None

# -------------------------------
# Initialization function
# -------------------------------
def init():
    global session, tokenizer, label_map
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../models/huk_distilbert.onnx')
        tokenizer_path = os.path.join(os.path.dirname(__file__), '../configs/tokenizer_files/')
        
        session = ort.InferenceSession(model_path)
        
        if os.path.exists(tokenizer_path):
            tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        print("✅ ONNX model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize ONNX model: {e}")

# -------------------------------
# Preprocess input text
# -------------------------------
def preprocess(text: str):
    global tokenizer
    encodings = tokenizer(
        text, 
        truncation=True, 
        padding='max_length', 
        max_length=256, 
        return_tensors='np'
    )
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    return input_ids, attention_mask

# -------------------------------
# Run inference
# -------------------------------
def run(raw_data):
    global session, label_map
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data
        
        text = data.get('text', '')
        input_ids, attention_mask = preprocess(text)
        ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        ort_outs = session.run(None, ort_inputs)
        
        logits = ort_outs[0]
        pred_id = int(np.argmax(logits, axis=1)[0])
        pred_label = label_map.get(pred_id, str(pred_id))
        confidence = float(np.max(logits))
        
        return json.dumps({'prediction': pred_label, 'score': confidence})
    except Exception as e:
        return json.dumps({'error': str(e)})
