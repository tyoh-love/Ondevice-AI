
# 내 시스템 정보 확인하기
import platform
import subprocess
import sys

def check_system_info():
    """시스템 정보를 친절하게 알려주는 함수"""
    
    print("🖥️ 시스템 정보 확인 중...")
    print("=" * 50)
    
    # 운영체제 확인
    os_info = platform.system()
    os_version = platform.version()
    print(f"📌 운영체제: {os_info} {os_version}")
    
    # Python 버전 확인
    python_version = sys.version.split()[0]
    print(f"🐍 Python 버전: {python_version}")
    
    # CPU 정보
    processor = platform.processor()
    print(f"💻 프로세서: {processor}")
    
    # GPU 확인 (NVIDIA)
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                           encoding='utf-8').strip()
        print(f"🎮 NVIDIA GPU: {nvidia_smi}")
    except:
        print("🎮 NVIDIA GPU: 감지되지 않음")
    
    # 메모리 확인
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 메모리: {memory.total / (1024**3):.1f}GB 중 {memory.available / (1024**3):.1f}GB 사용 가능")
    except:
        print("💾 메모리: psutil을 설치하면 확인 가능합니다")
    
    print("=" * 50)
    
    # 추천사항 제공
    if 'NVIDIA' in str(processor) or 'nvidia_smi' in locals():
        print("✅ GPU가 감지되었습니다! 빠른 AI 학습이 가능합니다.")
    else:
        print("💡 GPU가 없어도 괜찮습니다. CPU로도 충분히 실습 가능합니다!")
    
    if float(python_version[:3]) < 3.8:
        print("⚠️ Python 3.8 이상으로 업그레이드를 권장합니다.")
    else:
        print("✅ Python 버전이 적절합니다.")

# 시스템 정보 확인 실행
check_system_info()
