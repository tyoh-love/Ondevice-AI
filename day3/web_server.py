from flask import Flask, request, jsonify, send_from_directory, Response
import whisper
import numpy as np
import wave
import tempfile
import os
import torch
import ollama
import traceback
import edge_tts
import asyncio
import io
import uuid
import base64
from PIL import Image

app = Flask(__name__)

# Helper function to run async functions in Flask
def run_async(func):
    """Flask에서 async 함수를 실행하기 위한 헬퍼"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(func)
    finally:
        loop.close()

class Qwen2VLService:
    """Qwen2.5-VL Multimodal Service for both text and vision"""
    
    def __init__(self, model_name="qwen2.5-vl:latest"):
        self.model_name = model_name
        print("🤖 Qwen2.5-VL 멀티모달 서비스 초기화 중...")
    
    def answer_question(self, question, language="ko", image_data=None):
        """질문에 대한 답변 생성 (텍스트 또는 비전+텍스트)"""
        try:
            if image_data:
                print(f"👁️ Qwen2.5-VL로 비전+텍스트 질문 처리 중: {question}")
            else:
                print(f"🧠 Qwen2.5-VL로 텍스트 질문 처리 중: {question}")
            
            if language == "ko":
                if image_data:
                    prompt = f"""이미지를 보고 사용자의 질문에 대해 따뜻하고 상세한 답변을 한국어로 제공해주세요. 이미지에서 보이는 것을 정확히 설명하고 질문에 답해주세요.

질문: {question}

답변:"""
                else:
                    prompt = f"""사용자의 질문에 대해서 따뜻한 답변을 한국어로 제공해주세요. 간결하면서도 완전한 답변을 해주세요.

질문: {question}

답변:"""
            else:
                if image_data:
                    prompt = f"""Please analyze the image and provide an accurate, detailed answer to the user's question. Describe what you see and answer the question thoroughly.

Question: {question}

Answer:"""
                else:
                    prompt = f"""Please provide an accurate and helpful answer to the following question. Keep it concise but complete.

Question: {question}

Answer:"""
            
            # Prepare message content
            if image_data:
                # For vision tasks, include both text and image
                message_content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            else:
                # For text-only tasks
                message_content = prompt
            
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': message_content
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
            if image_data:
                print(f"✅ Qwen2.5-VL 비전 답변 완료")
            else:
                print(f"✅ Qwen2.5-VL 텍스트 답변 완료")
            return answer
            
        except Exception as e:
            print(f"❌ Qwen2.5-VL 처리 오류: {e}")
            traceback.print_exc()
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

class TTSService:
    """Edge-TTS Text-to-Speech Service"""
    
    def __init__(self):
        self.korean_voices = {
            'female': 'ko-KR-SunHiNeural',
            'male': 'ko-KR-InJoonNeural'
        }
        self.english_voices = {
            'female': 'en-US-JennyNeural', 
            'male': 'en-US-GuyNeural'
        }
        self.audio_cache = {}  # Simple in-memory cache
        print("🔊 TTS 서비스 초기화 중...")
    
    async def text_to_speech(self, text, language="ko", gender="female", rate="+0%", pitch="+0Hz"):
        """텍스트를 음성으로 변환"""
        try:
            print(f"🎵 TTS 변환 중: {text[:50]}...")
            
            # Voice selection based on language and gender
            if language == "ko":
                voice = self.korean_voices.get(gender, self.korean_voices['female'])
            else:
                voice = self.english_voices.get(gender, self.english_voices['female'])
            
            # Create cache key (simplified without rate/pitch)
            cache_key = f"{hash(text)}_{voice}"
            
            # Check cache first
            if cache_key in self.audio_cache:
                print("📦 캐시에서 오디오 반환")
                return self.audio_cache[cache_key]
            
            # Use plain text directly (no SSML to avoid XML tags being spoken)
            # Edge-TTS works best with simple text input
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            # Cache the result (limit cache size)
            if len(self.audio_cache) > 50:  # Limit cache size
                # Remove oldest entry
                oldest_key = next(iter(self.audio_cache))
                del self.audio_cache[oldest_key]
            
            self.audio_cache[cache_key] = audio_data
            
            print(f"✅ TTS 변환 완료 ({len(audio_data)} bytes)")
            return audio_data
            
        except Exception as e:
            print(f"❌ TTS 변환 오류: {e}")
            traceback.print_exc()
            return None
    
    def get_available_voices(self):
        """사용 가능한 음성 목록 반환"""
        return {
            'korean': self.korean_voices,
            'english': self.english_voices
        }
    
    async def text_to_speech_sync(self, text, language="ko", gender="female", rate="+0%", pitch="+0Hz"):
        """동기 래퍼 함수"""
        return await self.text_to_speech(text, language, gender, rate, pitch)

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
qa_service = Qwen2VLService()
tts_service = TTSService()

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
        tts_enabled = request.form.get('tts_enabled', 'true').lower() == 'true'
        tts_gender = request.form.get('tts_gender', 'female')
        
        print(f"🎙️ 음성 대화 요청: 언어={language}, 모델={model_size}, TTS={tts_enabled}")
        
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
        
        # 3단계: TTS 변환 (선택적)
        audio_url = None
        if tts_enabled:
            try:
                audio_data_tts = run_async(tts_service.text_to_speech(answer, language, tts_gender))
                if audio_data_tts:
                    # 오디오 ID 생성 및 임시 저장
                    audio_id = str(uuid.uuid4())
                    temp_dir = tempfile.gettempdir()
                    audio_path = os.path.join(temp_dir, f"tts_{audio_id}.mp3")
                    
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data_tts)
                    
                    audio_url = f'/audio/{audio_id}'
                    print(f"🔊 TTS 오디오 생성 완료: {audio_url}")
            except Exception as e:
                print(f"⚠️ TTS 변환 오류 (계속 진행): {e}")
        
        return jsonify({
            'success': True,
            'question': text,
            'answer': answer,
            'language': language,
            'model': model_size,
            'audio_url': audio_url,
            'tts_enabled': tts_enabled
        })
        
    except Exception as e:
        print(f"음성 대화 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/speak', methods=['POST'])
def speak():
    """텍스트를 음성으로 변환하는 API"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': '텍스트가 없습니다'})
        
        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': '빈 텍스트입니다'})
        
        language = data.get('language', 'ko')
        gender = data.get('gender', 'female')
        rate = data.get('rate', '+0%')
        pitch = data.get('pitch', '+0Hz')
        
        print(f"🔊 TTS 요청: {text[:50]}... (언어: {language}, 성별: {gender})")
        
        # TTS 변환
        audio_data = run_async(tts_service.text_to_speech(text, language, gender, rate, pitch))
        
        if audio_data is None:
            return jsonify({'success': False, 'error': 'TTS 변환 실패'})
        
        # 오디오 ID 생성 (임시 저장용)
        audio_id = str(uuid.uuid4())
        
        # 임시 파일로 저장
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"tts_{audio_id}.mp3")
        
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
        
        return jsonify({
            'success': True,
            'audio_id': audio_id,
            'audio_url': f'/audio/{audio_id}',
            'text': text,
            'language': language,
            'gender': gender
        })
        
    except Exception as e:
        print(f"TTS API 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/audio/<audio_id>')
def serve_audio(audio_id):
    """생성된 오디오 파일 제공"""
    try:
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"tts_{audio_id}.mp3")
        
        if not os.path.exists(audio_path):
            return "Audio not found", 404
        
        def generate():
            with open(audio_path, 'rb') as f:
                data = f.read()
                yield data
            # 파일 전송 후 삭제
            try:
                os.unlink(audio_path)
            except:
                pass
        
        return Response(generate(), mimetype='audio/mpeg')
        
    except Exception as e:
        print(f"오디오 서빙 오류: {e}")
        return "Error serving audio", 500

@app.route('/voices')
def get_voices():
    """사용 가능한 TTS 음성 목록 반환"""
    try:
        voices = tts_service.get_available_voices()
        return jsonify({
            'success': True,
            'voices': voices
        })
    except Exception as e:
        print(f"음성 목록 오류: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """이미지 분석 API"""
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'question' not in data:
            return jsonify({'success': False, 'error': '이미지와 질문이 필요합니다'})
        
        image_data = data['image']
        question = data['question'].strip()
        language = data.get('language', 'ko')
        
        if not question:
            question = "이 이미지에 대해 설명해주세요." if language == 'ko' else "Please describe this image."
        
        print(f"🖼️ 이미지 분석 요청: {question}")
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Generate answer using Qwen2.5-VL with vision
        answer = qa_service.answer_question(question, language, image_data)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'language': language
        })
        
    except Exception as e:
        print(f"이미지 분석 오류: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/vision_chat', methods=['POST'])
def vision_chat():
    """비전 + 음성 대화 API (STT + 이미지 분석 + TTS 통합)"""
    try:
        # 파일 및 데이터 확인
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': '오디오 파일이 없습니다'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': '파일이 선택되지 않았습니다'})
        
        # 이미지 데이터 확인
        image_data = request.form.get('image')
        if not image_data:
            return jsonify({'success': False, 'error': '이미지 데이터가 없습니다'})
        
        # 설정 가져오기
        language = request.form.get('language', 'ko')
        model_size = request.form.get('model', 'base')
        tts_enabled = request.form.get('tts_enabled', 'true').lower() == 'true'
        tts_gender = request.form.get('tts_gender', 'female')
        
        print(f"👁️🎙️ 비전 + 음성 대화 요청: 언어={language}, TTS={tts_enabled}")
        
        # 1단계: 음성 인식
        audio_data = stt_service.process_webm_audio(audio_file)
        if audio_data is None:
            return jsonify({'success': False, 'error': '오디오 처리 실패'})
        
        text = stt_service.transcribe_audio(audio_data, language, model_size)
        if text is None:
            return jsonify({'success': False, 'error': '음성 인식 실패'})
        
        print(f"📝 인식된 질문: {text}")
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 2단계: Qwen2.5-VL로 비전 분석
        answer = qa_service.answer_question(text, language, image_data)
        
        print(f"🤖 생성된 답변: {answer}")
        
        # 3단계: TTS 변환 (선택적)
        audio_url = None
        if tts_enabled:
            try:
                audio_data_tts = run_async(tts_service.text_to_speech(answer, language, tts_gender))
                if audio_data_tts:
                    # 오디오 ID 생성 및 임시 저장
                    audio_id = str(uuid.uuid4())
                    temp_dir = tempfile.gettempdir()
                    audio_path = os.path.join(temp_dir, f"tts_{audio_id}.mp3")
                    
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data_tts)
                    
                    audio_url = f'/audio/{audio_id}'
                    print(f"🔊 TTS 오디오 생성 완료: {audio_url}")
            except Exception as e:
                print(f"⚠️ TTS 변환 오류 (계속 진행): {e}")
        
        return jsonify({
            'success': True,
            'question': text,
            'answer': answer,
            'language': language,
            'model': model_size,
            'audio_url': audio_url,
            'tts_enabled': tts_enabled,
            'has_vision': True
        })
        
    except Exception as e:
        print(f"비전 + 음성 대화 오류: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    """서비스 상태 확인"""
    # Qwen2.5-VL 모델 상태 확인
    qwen2vl_available = False
    try:
        models_response = ollama.list()
        model_names = [model.model for model in models_response.models]
        qwen2vl_available = 'qwen2.5-vl:latest' in model_names
    except:
        pass
    
    return jsonify({
        'status': 'running',
        'loaded_models': list(stt_service.models.keys()),
        'available_models': ['base', 'small', 'medium', 'large'],
        'cuda_available': torch.cuda.is_available(),
        'qwen2vl_available': qwen2vl_available,
        'qa_service': 'Qwen2.5-VL Multimodal',
        'vision_enabled': True,
        'vad_enabled': True,
        'tts_service': 'Edge-TTS',
        'tts_voices': tts_service.get_available_voices()
    })

@app.route('/health')
def health():
    """헬스 체크"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Multimodal AI Assistant with Qwen2.5-VL 서버 시작 중...")
    print("📱 브라우저에서 http://localhost:5000 접속")
    print("🎤 마이크 권한을 허용해주세요 (VAD 자동 감지)")
    print("👁️ 카메라 권한을 허용해주세요 (비전 분석)")
    print("🤖 음성+비전으로 질문하면 Qwen2.5-VL이 답변합니다")
    print("🔊 TTS로 답변을 음성으로 들을 수 있습니다")
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