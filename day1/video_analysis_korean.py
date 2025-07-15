#!/usr/bin/env python3
# encoding: utf-8

"""
Video Analysis with Korean Summarization
Simplified CLI version that processes video files using MiniCPM-o2.6 and ExaOne3.5
"""

import argparse
import sys
import os
import io
import base64
import traceback
from PIL import Image
from decord import VideoReader
import ollama

# Constants
ERROR_MSG = "Error occurred during processing"
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}

def get_file_extension(filename):
    """Get file extension in lowercase"""
    return os.path.splitext(filename)[1].lower()

def is_video(filename):
    """Check if file is a video"""
    return get_file_extension(filename) in VIDEO_EXTENSIONS

def process_video_file(video_path):
    """Extract key frames from video file"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not is_video(video_path):
        raise ValueError(f"File is not a supported video format: {video_path}")
    
    print(f"Processing video: {video_path}")
    
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
        
        # Extract 4 key frames for efficiency
        max_frames = min(4, total_frames)
        
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // max_frames
            frame_indices = list(range(0, total_frames, step))[:max_frames]
        
        frames = []
        for idx in frame_indices:
            frame_array = vr[idx].asnumpy()
            frame_image = Image.fromarray(frame_array).convert("RGB")
            frames.append(frame_image)
        
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    except Exception as e:
        print(f"Error processing video: {e}")
        raise

def analyze_with_minicpm(frames, prompt="Analyze this video and describe what you see in detail."):
    """Analyze video frames using MiniCPM-o2.6 model"""
    try:
        print("Analyzing video with MiniCPM-o2.6...")
        
        # Convert frames to base64 for Ollama
        images = []
        for frame in frames:
            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_base64)
        
        # Prepare message for Ollama
        message = {
            'role': 'user',
            'content': prompt,
            'images': images
        }
        
        # Call MiniCPM model
        response = ollama.chat(
            model='openbmb/minicpm-o2.6:latest',
            messages=[message],
            options={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 100,
                'num_predict': 2048,
                'repeat_penalty': 1.05,
                'num_ctx': 8192
            }
        )
        
        analysis_result = response['message']['content']
        print("MiniCPM analysis completed")
        return analysis_result
    
    except Exception as e:
        print(f"Error in MiniCPM analysis: {e}")
        traceback.print_exc()
        return ERROR_MSG

def summarize_in_korean(english_text):
    """Summarize English text in Korean using ExaOne3.5 model"""
    try:
        print("Generating Korean summary with ExaOne3.5...")
        
        korean_prompt = f"""다음 영어 텍스트를 한국어로 요약해주세요. 핵심 내용을 간결하고 명확하게 정리해주세요.

영어 텍스트:
{english_text}

한국어 요약:"""
        
        # Call ExaOne model
        response = ollama.chat(
            model='exaone3.5:2.4b',
            messages=[{
                'role': 'user',
                'content': korean_prompt
            }],
            options={
                'temperature': 0.3,
                'top_p': 0.9,
                'num_predict': 1024,
                'repeat_penalty': 1.1,
                'num_ctx': 4096
            }
        )
        
        korean_summary = response['message']['content']
        print("Korean summary completed")
        return korean_summary
    
    except Exception as e:
        print(f"Error in Korean summarization: {e}")
        traceback.print_exc()
        return ERROR_MSG

def test_ollama_connection():
    """Test connection to Ollama and verify models are available"""
    try:
        models_response = ollama.list()
        model_names = [model.model for model in models_response.models]
        
        required_models = ['openbmb/minicpm-o2.6:latest', 'exaone3.5:2.4b']
        missing_models = [model for model in required_models if model not in model_names]
        
        if missing_models:
            print(f"Missing required models: {missing_models}")
            print("Please pull the required models:")
            for model in missing_models:
                print(f"  ollama pull {model}")
            return False
        
        print("All required models are available")
        return True
    
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        return False

def save_results(output_path, original_analysis, korean_summary):
    """Save analysis results to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== Video Analysis Results ===\n\n")
            f.write("Original Analysis (English):\n")
            f.write("-" * 40 + "\n")
            f.write(original_analysis + "\n\n")
            f.write("Korean Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(korean_summary + "\n")
        
        print(f"Results saved to: {output_path}")
    
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Video Analysis with Korean Summarization')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--prompt', default="Analyze this video and describe what you see in detail.", 
                       help='Custom prompt for video analysis')
    
    args = parser.parse_args()
    
    # Test Ollama connection
    if not test_ollama_connection():
        sys.exit(1)
    
    try:
        # Process video
        frames = process_video_file(args.video)
        
        # Analyze with MiniCPM
        analysis_result = analyze_with_minicpm(frames, args.prompt)
        
        if analysis_result == ERROR_MSG:
            print("Failed to analyze video with MiniCPM")
            sys.exit(1)
        
        # Summarize in Korean
        korean_summary = summarize_in_korean(analysis_result)
        
        if korean_summary == ERROR_MSG:
            print("Failed to generate Korean summary")
            sys.exit(1)
        
        # Display results
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print("\nOriginal Analysis (English):")
        print("-" * 40)
        print(analysis_result)
        print("\nKorean Summary:")
        print("-" * 40)
        print(korean_summary)
        
        # Save results if output path provided
        if args.output:
            save_results(args.output, analysis_result, korean_summary)
        
        print("\nProcessing completed successfully!")
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()