# MiniCPM-O Ollama Integration

This is an updated version of the MiniCPM-O demo that uses **Ollama** instead of HuggingFace for much faster loading and reduced memory usage.

## Prerequisites

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull the MiniCPM-O model**:
   ```bash
   ollama pull openbmb/minicpm-o2.6:latest
   ```

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements-ollama.txt
   ```

2. **Test the integration**:
   ```bash
   python test_ollama_integration.py
   ```

## Usage

1. **Run the demo**:
   ```bash
   python miniCPM-ollama.py
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:8000`

## Key Benefits

- **Faster startup**: No need to load large PyTorch models into memory
- **Lower memory usage**: Ollama manages the model efficiently
- **Better stability**: No GPU memory management issues
- **Easier deployment**: Works out of the box with Ollama
- **Simplified code**: Removed complex preprocessing - Ollama handles it
- **Fewer dependencies**: No PyTorch, transformers, or accelerate needed

## Features

- ✅ Chat with single images
- ✅ Chat with multiple images  
- ✅ Chat with videos
- ✅ In-context few-shot learning
- ✅ Gradio web interface
- ✅ Ollama-based inference

## Model Information

- **Model**: `openbmb/minicpm-o2.6:latest`
- **Size**: ~5.5 GB (quantized)
- **Format**: GGUF
- **Family**: Qwen2 + CLIP vision

## Troubleshooting

If you encounter issues:

1. **Ensure Ollama is running**: `ollama serve`
2. **Check model availability**: `ollama list`
3. **Test basic functionality**: `python test_ollama_integration.py`
4. **Check logs**: Look at the console output for error messages