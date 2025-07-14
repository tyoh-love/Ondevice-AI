#!/usr/bin/env python3

"""
Simple test script to verify Ollama integration works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import ollama
    print("✓ Ollama import successful")
    
    # Test connection
    models = ollama.list()
    print(f"✓ Ollama connection successful, {len(models['models'])} models available")
    
    # Check if our model is available
    model_name = "openbmb/minicpm-o2.6:latest"
    model_found = False
    print(f"Available models: {models}")
    
    if 'models' in models:
        for model in models['models']:
            print(f"Checking model: {model}")
            if hasattr(model, 'model') and model.model == model_name:
                model_found = True
                print(f"✓ Model {model_name} found")
                break
    else:
        print("No 'models' key found in response")
    
    if not model_found:
        print(f"✗ Model {model_name} not found")
        sys.exit(1)
    
    # Test basic text chat
    print("Testing basic text chat...")
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': 'Hello, can you see me?'}],
        options={'temperature': 0.7, 'num_predict': 50}
    )
    print(f"✓ Basic chat test successful: {response['message']['content'][:50]}...")
    
    print("\n🎉 All tests passed! Ollama integration is working correctly.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install ollama: pip install ollama")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)