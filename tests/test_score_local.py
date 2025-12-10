# tests/test_score_local.py
# Local ONNX inference test for AWS

import json
from aws.score import init, run

def test_score():
    # Initialize ONNX session and tokenizer
    init()
    
    # Sample input text
    sample = {'text': 'The insurance claim process was fast and easy.'}
    
    # Run inference
    result = run(json.dumps(sample))
    
    print('✅ AWS ONNX inference result:', result)

if __name__ == '__main__':
    test_score()
