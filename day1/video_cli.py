#!/usr/bin/env python3
"""
Command-line interface for video analysis using MiniCPM-O with Ollama

Usage:
    python video_cli.py --video sample_video.mp4 --question "What's happening?"
    python video_cli.py -v /path/to/video.mp4 -q "Describe the main objects"
"""

import argparse
import sys
import os
import io
import base64
from PIL import Image
from decord import VideoReader
import ollama

def check_ollama_connection(model_name):
    """Check if Ollama is running and model is available"""
    try:
        models = ollama.list()
        print(f"✓ Connected to Ollama ({len(models.get('models', []))} models available)")
        
        # Check if specific model is available
        model_found = any(
            model.get('model') == model_name or model.get('name') == model_name
            for model in models.get('models', [])
        )
        
        if not model_found:
            print(f"⚠️  Model '{model_name}' not found. Available models:")
            for model in models.get('models', []):
                print(f"   - {model.get('model', model.get('name', 'Unknown'))}")
            print(f"\nTo install the model, run: ollama pull {model_name}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False

def process_video_frames(video_path, max_frames=4):
    """Extract key frames from video for analysis"""
    print(f"📹 Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
        print(f"   Total frames: {total_frames}")
        
        # Extract key frames
        max_frames = min(max_frames, total_frames)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // max_frames
            frame_indices = list(range(0, total_frames, step))[:max_frames]
        
        print(f"   Extracting {len(frame_indices)} key frames: {frame_indices}")
        
        frames = []
        for idx in frame_indices:
            frame_array = vr[idx].asnumpy()
            frame_image = Image.fromarray(frame_array).convert("RGB")
            frames.append(frame_image)
        
        print(f"✓ Successfully extracted {len(frames)} frames")
        return frames
        
    except Exception as e:
        raise RuntimeError(f"Error processing video: {e}")

def frames_to_base64(frames):
    """Convert PIL Images to base64 for Ollama"""
    base64_images = []
    for frame in frames:
        buffered = io.BytesIO()
        frame.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        base64_images.append(img_base64)
    return base64_images

def analyze_video_with_ollama(frames, question, model_name):
    """Send frames and question to Ollama for analysis"""
    print(f"🤖 Analyzing video with question: '{question}'")
    
    # Convert frames to base64
    base64_images = frames_to_base64(frames)
    
    # Prepare message for Ollama
    message = {
        'role': 'user',
        'content': question,
        'images': base64_images
    }
    
    # Chat options
    options = {
        'temperature': 0.7,
        'top_p': 0.8,
        'top_k': 100,
        'num_predict': 512,
        'repeat_penalty': 1.05,
        'num_ctx': 8192
    }
    
    try:
        print("   Sending request to Ollama...")
        response = ollama.chat(
            model=model_name,
            messages=[message],
            options=options
        )
        
        answer = response['message']['content']
        print("✓ Analysis complete!")
        return answer
        
    except Exception as e:
        raise RuntimeError(f"Error during analysis: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze videos using MiniCPM-O with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_cli.py --video sample.mp4
  python video_cli.py -v /path/to/video.mp4 -q "What objects do you see?"
  python video_cli.py -v video.mp4 -q "Describe the activities" --model openbmb/minicpm-o2.6:latest
        """
    )
    
    parser.add_argument(
        '-v', '--video',
        required=True,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '-q', '--question',
        default="What's happening in this video? Describe what you see.",
        help='Question to ask about the video (default: general description)'
    )
    
    parser.add_argument(
        '--model',
        default="openbmb/minicpm-o2.6:latest",
        help='Ollama model name (default: openbmb/minicpm-o2.6:latest)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=4,
        help='Maximum number of frames to extract (default: 4)'
    )
    
    args = parser.parse_args()
    
    print("🎬 MiniCPM-O Video Analysis CLI")
    print("=" * 50)
    
    try:
        # Check Ollama connection and model availability
        if not check_ollama_connection(args.model):
            sys.exit(1)
        
        # Process video frames
        frames = process_video_frames(args.video, args.frames)
        
        # Analyze with Ollama
        result = analyze_video_with_ollama(frames, args.question, args.model)
        
        # Display results
        print("\n" + "=" * 50)
        print("📋 ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Video: {args.video}")
        print(f"Question: {args.question}")
        print(f"Frames analyzed: {len(frames)}")
        print("-" * 50)
        print(result)
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()