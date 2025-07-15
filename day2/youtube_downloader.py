#!/usr/bin/env python3
"""
Simple YouTube Video Downloader using yt-dlp
Usage: python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"
"""

import sys
import os
import yt_dlp


def download_video(url, output_path=None):
    """
    Download a YouTube video from the given URL using yt-dlp
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the video (default: current directory)
    
    Returns:
        str: Path to the downloaded file
    """
    try:
        # Set output path
        if output_path is None:
            output_path = os.getcwd()
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer MP4 format
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),  # Output template
            'quiet': False,  # Show download progress
            'no_warnings': False,
            'progress_hooks': [download_progress_hook],
        }
        
        # Create yt-dlp object and download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Fetching video information from: {url}")
            
            # Extract video info first
            info = ydl.extract_info(url, download=False)
            
            # Display video information
            print(f"\nTitle: {info.get('title', 'N/A')}")
            print(f"Uploader: {info.get('uploader', 'N/A')}")
            print(f"Duration: {info.get('duration', 0)} seconds")
            print(f"Views: {info.get('view_count', 0):,}")
            
            # Get format info
            format_info = info.get('format', 'N/A')
            if 'height' in info:
                print(f"Resolution: {info.get('height', 'N/A')}p")
            
            # Download the video
            print(f"\nDownloading to: {output_path}")
            ydl.download([url])
            
            # Get the filename
            filename = ydl.prepare_filename(info)
            # Replace extension if needed
            if not filename.endswith('.mp4'):
                base_name = os.path.splitext(filename)[0]
                filename = f"{base_name}.mp4"
            
            full_path = os.path.join(output_path, os.path.basename(filename))
            
            print(f"\n✅ Download completed!")
            print(f"Saved as: {full_path}")
            
            return full_path
        
    except yt_dlp.utils.DownloadError as e:
        print(f"\n❌ Download error: {str(e)}")
        return None
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        return None


def download_progress_hook(d):
    """Progress hook for yt-dlp downloads"""
    if d['status'] == 'downloading':
        total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        downloaded = d.get('downloaded_bytes', 0)
        
        if total > 0:
            percent = (downloaded / total) * 100
            speed = d.get('speed', 0)
            if speed:
                speed_mb = speed / (1024 * 1024)
                print(f"\rDownloading: {percent:.1f}% | Speed: {speed_mb:.2f} MB/s", end='', flush=True)
            else:
                print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
    elif d['status'] == 'finished':
        print("\nProcessing downloaded file...")


def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <YouTube URL>")
        print("Example: python youtube_downloader.py \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\"")
        sys.exit(1)
    
    # Get YouTube URL from command line argument
    youtube_url = sys.argv[1]
    
    # Validate URL format
    if not youtube_url.startswith(('http://youtube.com/', 'https://youtube.com/', 
                                   'http://www.youtube.com/', 'https://www.youtube.com/',
                                   'http://youtu.be/', 'https://youtu.be/')):
        print("❌ Error: Please provide a valid YouTube URL")
        sys.exit(1)
    
    # Optional: Get output directory from second argument
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Download the video
    download_video(youtube_url, output_dir)


if __name__ == "__main__":
    main()
