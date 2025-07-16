# Whisper Web STT - WSL Ubuntu Compatible

Web-based Whisper speech-to-text interface that works without portaudio dependencies.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   sudo apt update && sudo apt install ffmpeg
   ```

2. **Run the web interface:**
   ```bash
   python run_web.py
   ```

3. **Open browser and navigate to:**
   ```
   http://localhost:5000
   ```

## 📁 Files

- `index.html` - Web UI with audio recording
- `web_server.py` - Flask backend for audio processing  
- `run_web.py` - Easy launcher script
- `requirements.txt` - Python dependencies

## ✨ Features

- **No portaudio needed** - Uses Web Audio API
- **Real-time visualization** - Audio level bars
- **Multiple languages** - Korean, English, auto-detect
- **Model selection** - Base, Small, Medium, Large
- **WSL compatible** - Works in Windows Subsystem for Linux

## 🎤 Usage

1. Click "녹음 시작" to start recording
2. Speak into your microphone
3. Click "녹음 중지" to process audio
4. View transcription results below

## 🛠️ Dependencies

- Flask - Web framework
- OpenAI Whisper - Speech recognition
- FFmpeg - Audio processing
- WebRTC VAD - Voice activity detection

## 💡 Benefits over original

- No sounddevice/portaudio issues in WSL
- Browser-based - works anywhere
- Better cross-platform compatibility
- Modern web interface
- Real-time audio visualization