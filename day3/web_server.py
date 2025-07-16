from flask import Flask, request, jsonify, send_from_directory
import whisper
import numpy as np
import wave
import tempfile
import os
import torch

app = Flask(__name__)

class WebWhisperSTT:
    """Web-based Whisper STT Service"""
    
    def __init__(self):
        self.models = {}
        print("🌐 Web Whisper STT 서비스 초기화 중...")
    
    def load_model(self, model_size="base"):
        """모델 로드 (캐싱)"""
        if model_size not in self.models:
            print(f"📥 Whisper {model_size} 모델 로드 중...")
            self.models[model_size] = whisper.load_model(model_size)
            print(f"✅ {model_size} 모델 로드 완료")
        return self.models[model_size]
    
    def process_webm_audio(self, audio_file):
        """WebM 오디오를 numpy 배열로 변환"""
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                audio_file.save(temp_file.name)
                temp_path = temp_file.name
            
            # ffmpeg를 사용하여 WAV로 변환
            import subprocess
            wav_path = temp_path.replace('.webm', '.wav')
            
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_path, 
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                wav_path
            ], capture_output=True, check=True)
            
            # WAV 파일 읽기
            with wave.open(wav_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sound_info = np.frombuffer(frames, dtype=np.int16)
                audio_float32 = sound_info.astype(np.float32) / 32768.0
            
            # 임시 파일 정리
            os.unlink(temp_path)
            os.unlink(wav_path)
            
            return audio_float32
            
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 오류: {e}")
            return None
        except Exception as e:
            print(f"오디오 처리 오류: {e}")
            return None
    
    def transcribe_audio(self, audio_data, language="ko", model_size="base"):
        """음성을 텍스트로 변환"""
        try:
            model = self.load_model(model_size)
            
            # 최소 길이 확보
            if len(audio_data) < 16000:  # 1초 미만
                audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)))
            
            # Whisper 실행
            result = model.transcribe(
                audio_data,
                language=None if language == "auto" else language,
                fp16=torch.cuda.is_available()
            )
            
            return result["text"].strip()
            
        except Exception as e:
            print(f"음성 인식 오류: {e}")
            return None

# 전역 STT 인스턴스
stt_service = WebWhisperSTT()

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """음성 인식 API"""
    try:
        # 파일 확인
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '오디오 파일이 없습니다'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다'})
        
        # 설정 가져오기
        language = request.form.get('language', 'ko')
        model_size = request.form.get('model', 'base')
        
        print(f"🎤 음성 인식 요청: 언어={language}, 모델={model_size}")
        
        # 오디오 처리
        audio_data = stt_service.process_webm_audio(audio_file)
        if audio_data is None:
            return jsonify({'success': False, 'error': '오디오 처리 실패'})
        
        # 음성 인식
        text = stt_service.transcribe_audio(audio_data, language, model_size)
        if text is None:
            return jsonify({'success': False, 'error': '음성 인식 실패'})
        
        print(f"📝 인식 결과: {text}")
        
        return jsonify({
            'success': True,
            'text': text,
            'language': language,
            'model': model_size
        })
        
    except Exception as e:
        print(f"서버 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    """서비스 상태 확인"""
    return jsonify({
        'status': 'running',
        'loaded_models': list(stt_service.models.keys()),
        'available_models': ['base', 'small', 'medium', 'large'],
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Whisper Web STT 서버 시작 중...")
    print("📱 브라우저에서 http://localhost:5000 접속")
    print("🎤 마이크 권한을 허용해주세요")
    print("⚡ CUDA 사용 가능:", torch.cuda.is_available())
    
    # FFmpeg 설치 확인
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg 설치 확인됨")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg가 설치되지 않았습니다. 설치가 필요합니다:")
        print("   Ubuntu/WSL: sudo apt install ffmpeg")
        print("   Windows: chocolatey로 설치하거나 공식 사이트에서 다운로드")
    
    app.run(host='0.0.0.0', port=5000, debug=True)