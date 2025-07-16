#!/usr/bin/env python3
"""
Web-based Whisper STT runner - no portaudio needed!
WSL Ubuntu compatible version using web interface
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """필요한 의존성 확인"""
    print("🔍 의존성 확인 중...")
    
    # Python 패키지 확인
    required_packages = [
        'flask', 'whisper', 'numpy', 'torch', 'webrtcvad'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령으로 설치하세요:")
        print("pip install -r requirements.txt")
        return False
    
    # FFmpeg 확인
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg - 설치 필요")
        print("Ubuntu/WSL: sudo apt update && sudo apt install ffmpeg")
        return False
    
    return True

def run_server():
    """웹 서버 실행"""
    print("\n🚀 Whisper Web STT 서버 시작...")
    
    # 현재 디렉토리에서 실행
    os.chdir(Path(__file__).parent)
    
    # 서버 시작
    try:
        from web_server import app
        print("📱 브라우저에서 http://localhost:5000 접속하세요")
        print("🎤 마이크 권한을 허용해주세요")
        print("💡 Ctrl+C로 종료")
        
        # 자동으로 브라우저 열기 (WSL에서는 작동하지 않을 수 있음)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
        
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\n👋 서버 종료")
    except Exception as e:
        print(f"❌ 서버 실행 오류: {e}")

def main():
    """메인 함수"""
    print("🎤 Whisper Web STT - WSL Ubuntu 호환 버전")
    print("=" * 50)
    
    if not check_dependencies():
        print("\n❌ 의존성 확인 실패. 설치 후 다시 실행하세요.")
        sys.exit(1)
    
    print("\n✅ 모든 의존성 확인 완료!")
    run_server()

if __name__ == "__main__":
    main()