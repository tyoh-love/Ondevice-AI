
import whisper
import numpy as np
import queue
import threading
import webrtcvad
import collections
import time
from typing import Optional, Callable
import matplotlib.pyplot as plt
import torch

class AdvancedWhisperSTT:
    """고급 음성 인식 시스템"""
    
    def __init__(self, model_size="base", language="ko"):
        print(f"🎤 Whisper {model_size} 모델을 로드하고 있습니다...")
        
        self.model = whisper.load_model(model_size)
        self.language = language
        
        # VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad(2)  # 민감도 0-3
        
        # 오디오 설정
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # 버퍼
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        print("✅ 음성 인식 시스템 준비 완료!")
    
    def visualize_audio_processing(self):
        """오디오 처리 과정 시각화"""
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # 1. 원본 오디오 신호
        ax1 = axes[0, 0]
        t = np.linspace(0, 1, self.sample_rate)
        # 음성 신호 시뮬레이션 (여러 주파수 혼합)
        signal = (0.5 * np.sin(2 * np.pi * 200 * t) + 
                 0.3 * np.sin(2 * np.pi * 400 * t) + 
                 0.2 * np.sin(2 * np.pi * 800 * t))
        noise = np.random.normal(0, 0.1, len(t))
        audio = signal + noise
        
        ax1.plot(t[:1000], audio[:1000], alpha=0.7)
        ax1.set_title('원본 오디오 신호', fontweight='bold')
        ax1.set_xlabel('시간 (초)')
        ax1.set_ylabel('진폭')
        ax1.grid(True, alpha=0.3)
        
        # 2. 주파수 스펙트럼
        ax2 = axes[0, 1]
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        fft = np.abs(np.fft.fft(audio))
        
        ax2.plot(freqs[:len(freqs)//2], fft[:len(fft)//2])
        ax2.set_title('주파수 스펙트럼', fontweight='bold')
        ax2.set_xlabel('주파수 (Hz)')
        ax2.set_ylabel('강도')
        ax2.set_xlim(0, 2000)
        ax2.grid(True, alpha=0.3)
        
        # 3. VAD (음성 활동 감지)
        ax3 = axes[1, 0]
        # VAD 시뮬레이션
        vad_result = np.abs(audio) > 0.3
        vad_smoothed = np.convolve(vad_result, np.ones(100)/100, mode='same')
        
        ax3.plot(t[:5000], audio[:5000], alpha=0.5, label='오디오')
        ax3.fill_between(t[:5000], -1, 1, where=vad_smoothed[:5000] > 0.5,
                         alpha=0.3, color='red', label='음성 감지')
        ax3.set_title('음성 활동 감지 (VAD)', fontweight='bold')
        ax3.set_xlabel('시간 (초)')
        ax3.set_ylabel('진폭')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 멜 스펙트로그램
        ax4 = axes[1, 1]
        # 간단한 스펙트로그램 시뮬레이션
        spec_data = np.random.rand(128, 100) * np.linspace(1, 0.1, 128).reshape(-1, 1)
        im = ax4.imshow(spec_data, aspect='auto', origin='lower', cmap='viridis')
        ax4.set_title('멜 스펙트로그램', fontweight='bold')
        ax4.set_xlabel('시간 프레임')
        ax4.set_ylabel('멜 빈')
        plt.colorbar(im, ax=ax4)
        
        # 5. 음성 인식 과정
        ax5 = axes[2, 0]
        ax5.text(0.5, 0.8, '음성 입력', ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax5.text(0.5, 0.6, '↓', ha='center', fontsize=16)
        ax5.text(0.5, 0.4, '특징 추출', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        ax5.text(0.5, 0.2, '↓', ha='center', fontsize=16)
        ax5.text(0.5, 0.0, '텍스트 변환', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax5.set_xlim(0, 1)
        ax5.set_ylim(-0.2, 1)
        ax5.axis('off')
        ax5.set_title('Whisper 처리 과정', fontweight='bold')
        
        # 6. 언어 모델 확률
        ax6 = axes[2, 1]
        words = ['안녕하세요', '안녕하십니까', '안녕히', '안녕', '안년하세요']
        probs = [0.4, 0.3, 0.15, 0.1, 0.05]
        
        bars = ax6.bar(words, probs, color='orange')
        ax6.set_title('언어 모델 예측 확률', fontweight='bold')
        ax6.set_ylabel('확률')
        ax6.set_xticklabels(words, rotation=45, ha='right')
        
        for bar, prob in zip(bars, probs):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.2f}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def voice_activity_detection(self, audio_frame):
        """음성 활동 감지"""
        # 16비트 PCM으로 변환
        audio_int16 = (audio_frame * 32767).astype(np.int16)
        
        # VAD 적용
        return self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
    
    def continuous_recognition(self, callback: Callable[[str], None]):
        """연속 음성 인식"""
        
        def audio_callback(indata, frames, time_info, status):
            """오디오 콜백"""
            if status:
                print(f"오디오 상태: {status}")
            
            self.audio_queue.put(indata.copy())
        
        # 스트림 시작
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=self.frame_size
        )
        
        with stream:
            print("\n🎤 음성 인식을 시작합니다...")
            print("말씀하세요! (종료: Ctrl+C)")
            
            # 링 버퍼 (3초 분량)
            ring_buffer = collections.deque(maxlen=100)
            
            # 음성 감지 상태
            triggered = False
            voiced_frames = []
            
            while self.is_recording:
                try:
                    frame = self.audio_queue.get(timeout=0.1)
                    
                    # VAD 확인
                    is_speech = self.voice_activity_detection(frame)
                    
                    if not triggered:
                        ring_buffer.append((frame, is_speech))
                        num_voiced = len([f for f, speech in ring_buffer if speech])
                        
                        # 음성 시작 감지 (0.3초 이상)
                        if num_voiced > 0.3 * 100:
                            triggered = True
                            print("🔊 음성 감지됨...")
                            
                            # 링 버퍼의 내용을 voiced_frames에 추가
                            for f, s in ring_buffer:
                                voiced_frames.append(f)
                            ring_buffer.clear()
                    else:
                        # 음성 수집
                        voiced_frames.append(frame)
                        ring_buffer.append((frame, is_speech))
                        
                        # 음성 종료 감지 (1초 이상 조용)
                        num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                        if num_unvoiced > 1.0 * 100:
                            print("🔇 음성 종료, 인식 중...")
                            
                            # 음성 인식 수행
                            audio_data = np.concatenate(voiced_frames)
                            text = self.recognize(audio_data)
                            
                            if text and callback:
                                callback(text)
                            
                            # 초기화
                            triggered = False
                            voiced_frames = []
                            ring_buffer.clear()
                            
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    break
    
    def recognize(self, audio_data: np.ndarray) -> str:
        """음성을 텍스트로 변환"""
        # Whisper는 float32 필요
        audio_float32 = audio_data.astype(np.float32)
        
        # 패딩 (최소 길이 확보)
        if len(audio_float32) < self.sample_rate:
            audio_float32 = np.pad(audio_float32, (0, self.sample_rate - len(audio_float32)))
        
        # 음성 인식
        result = self.model.transcribe(
            audio_float32,
            language=self.language,
            fp16=torch.cuda.is_available()
        )
        
        return result["text"].strip()
    
    def start_recording(self):
        """녹음 시작"""
        self.is_recording = True
        
        def on_recognition(text):
            print(f"\n📝 인식된 텍스트: {text}")
        
        # 별도 스레드에서 실행
        self.recognition_thread = threading.Thread(
            target=self.continuous_recognition,
            args=(on_recognition,)
        )
        self.recognition_thread.start()
    
    def stop_recording(self):
        """녹음 중지"""
        self.is_recording = False
        if hasattr(self, 'recognition_thread'):
            self.recognition_thread.join()

# 음성 인식 시스템 테스트
def test_advanced_stt():
    stt = AdvancedWhisperSTT(model_size="base", language="ko")
    
    # 오디오 처리 과정 시각화
    stt.visualize_audio_processing()
    
    # 실시간 음성 인식 테스트
    print("\n=== 실시간 음성 인식 테스트 ===")
    
    try:
        stt.start_recording()
        time.sleep(30)  # 30초 동안 녹음
    except KeyboardInterrupt:
        print("\n인식 중단...")
    finally:
        stt.stop_recording()
        
if __name__ == "__main__":
    test_advanced_stt()
    