#!/usr/bin/env python
# encoding: utf-8

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os

# Model configuration
model_path = "openbmb/MiniCPM-o-2_6"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading MiniCPM-O-2.6 model...")
print(f"Device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    init_audio=False,  # Disable audio for image-only tasks
    init_tts=False     # Disable TTS for image-only tasks
)
model = model.to(device=device)
model.eval()

print("✅ Model loaded successfully!")

def analyze_image(image_path, question="What do you see in this image?"):
    """Analyze an image and answer questions about it"""
    
    # Load and process image
    print(f"\n📷 Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Prepare messages for the model
    msgs = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': question}
            ]
        }
    ]
    
    # Model parameters
    params = {
        'sampling': True,
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 100,
        'max_new_tokens': 512
    }
    
    print(f"❓ Question: {question}")
    print("🤔 Analyzing...")
    
    # Generate response
    with torch.no_grad():
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
    
    print(f"🤖 Answer: {answer}")
    return answer

# Main execution
if __name__ == "__main__":
    # Path to cry.png
    image_path = "../cry.png"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        print("Please make sure cry.png is in the parent directory of day1/")
        exit(1)
    
    print("\n" + "="*60)
    print("🎨 MiniCPM-O-2.6 Image Analysis Demo")
    print("="*60)
    
    # Test with multiple questions
    questions = [
        "What do you see in this image?",
        "Describe the emotions or mood in this image.",
        "What is the main subject or character in this image?",
        "What artistic style or technique is used in this image?",
        "What message or story does this image convey?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        answer = analyze_image(image_path, question)
        print("-"*60)