# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an on-device AI project focused on implementing and experimenting with AI models that can run locally. The codebase currently contains implementations for video understanding using MiniCPM-V model.

## Development Setup

### Dependencies Installation
```bash
pip install -r day1/requirements.txt
```

The requirements include:
- PyTorch ecosystem (torch, torchvision, transformers)
- Model optimization tools (accelerate, bitsandbytes)
- Image/video processing (opencv-python, PIL, supervision, decord)
- Audio processing (openai-whisper, gtts, pygame, sounddevice)
- Web interface tools (gradio, fastapi, uvicorn)

### Running Code

To run the video understanding AI:
```bash
cd day1
python miniCPM.py
```

To check system compatibility:
```bash
cd day1
python test.py
```

## Architecture Overview

### Core Components

1. **VideoUnderstandingAI** (day1/miniCPM.py:9-121)
   - Main class for video analysis using MiniCPM-V 2.6 model
   - Handles frame extraction, visualization, and question-answering about videos
   - Automatically detects and uses GPU if available

### Key Technical Details

- The project uses MiniCPM-V 2.6 model from `openbmb/MiniCPM-V-2_6`
- Supports both CPU and GPU execution with automatic device detection
- Uses bfloat16 precision on GPU for efficiency
- Extracts frames uniformly from videos for analysis

### Project Structure

```
ondevice-ai/
├── day1/
│   ├── miniCPM.py      # Video understanding AI implementation
│   ├── requirements.txt # Python dependencies
│   └── test.py         # System compatibility checker
└── 온디바이스AI강의.md   # Course materials (Korean)
```

## Important Notes

- The video AI expects a video file named "sample_video.mp4" in the working directory
- GPU (CUDA) is recommended but not required - the code falls back to CPU automatically
- The model uses `trust_remote_code=True` to allow custom model implementations