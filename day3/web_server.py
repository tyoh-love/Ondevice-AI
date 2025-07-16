from flask import Flask, request, jsonify, send_from_directory
import whisper
import numpy as np
import wave
import tempfile
import os
import torch
import ollama
import traceback

app = Flask(__name__)

class ExaOneService:
    """ExaOne Q&A Service"""
    
    def __init__(self, model_name="exaone3.5:2.4b"):
        self.model_name = model_name
        print("🤖 ExaOne Q&A 서비스 초기화 중...")
    
    def answer_question(self, question, language="ko"):
        """질문에 대한 답변 생성"""
        try:
            print(f"🧠 ExaOne으로 질문 처리 중: {question}")
            
            if language == "ko":
                prompt = f"""다음 질문에 대해 정확하고 도움이 되는 답변을 한국어로 제공해주세요. 간결하면서도 완전한 답변을 해주세요.

질문: {question}

답변:"""
            else:
                prompt = f"""Please provide an accurate and helpful answer to the following question. Keep it concise but complete.

Question: {question}

Answer:"""
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 1024,
                    'repeat_penalty': 1.1,
                    'num_ctx': 4096
                }
            )
            
            answer = response['message']['content'].strip()
            print(f"✅ ExaOne 답변 완료")
            return answer
            
        except Exception as e:
            print(f"❌ ExaOne 처리 오류: {e}")
            traceback.print_exc()
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

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

# 전역 서비스 인스턴스
stt_service = WebWhisperSTT()
qa_service = ExaOneService()

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

@app.route('/answer', methods=['POST'])
def answer_question():
    """ExaOne으로 질문 답변 API"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'error': '질문이 없습니다'})
        
        question = data['question'].strip()
        if not question:
            return jsonify({'success': False, 'error': '빈 질문입니다'})
        
        language = data.get('language', 'ko')
        
        print(f"💬 질문 처리 요청: {question}")
        
        # ExaOne으로 답변 생성
        answer = qa_service.answer_question(question, language)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'language': language
        })
        
    except Exception as e:
        print(f"질문 답변 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat', methods=['POST'])
def voice_chat():
    """음성 대화 API (STT + ExaOne 통합)"""
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
        
        print(f"🎙️ 음성 대화 요청: 언어={language}, 모델={model_size}")
        
        # 1단계: 음성 인식
        audio_data = stt_service.process_webm_audio(audio_file)
        if audio_data is None:
            return jsonify({'success': False, 'error': '오디오 처리 실패'})
        
        text = stt_service.transcribe_audio(audio_data, language, model_size)
        if text is None:
            return jsonify({'success': False, 'error': '음성 인식 실패'})
        
        print(f"📝 인식된 질문: {text}")
        
        # 2단계: ExaOne으로 답변 생성
        answer = qa_service.answer_question(text, language)
        
        print(f"🤖 생성된 답변: {answer}")
        
        return jsonify({
            'success': True,
            'question': text,
            'answer': answer,
            'language': language,
            'model': model_size
        })
        
    except Exception as e:
        print(f"음성 대화 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    """서비스 상태 확인"""
    # ExaOne 모델 상태 확인
    exaone_available = False
    try:
        models_response = ollama.list()
        model_names = [model.model for model in models_response.models]
        exaone_available = 'exaone3.5:2.4b' in model_names
    except:
        pass
    
    return jsonify({
        'status': 'running',
        'loaded_models': list(stt_service.models.keys()),
        'available_models': ['base', 'small', 'medium', 'large'],
        'cuda_available': torch.cuda.is_available(),
        'exaone_available': exaone_available,
        'qa_service': 'ExaOne Q&A',
        'vad_enabled': True
    })

@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Voice Q&A with ExaOne 서버 시작 중...")
    print("📱 브라우저에서 http://localhost:5000 접속")
    print("🎤 마이크 권한을 허용해주세요 (VAD 자동 감지)")
    print("🤖 음성으로 질문하면 ExaOne이 답변합니다")
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