# MiniCPM-V 2.6 실습
# 주의: 이 모델은 더 많은 메모리가 필요합니다

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import matplotlib.pyplot as plt

class VideoUnderstandingAI:
    """비디오를 이해하는 똑똑한 AI"""
    
    def __init__(self):
        """MiniCPM-V 2.6 모델 초기화"""
        print("🎬 비디오 이해 AI를 준비하고 있습니다...")
        
        model_name = "openbmb/MiniCPM-V-2_6"
        
        # 토크나이저: 텍스트를 토큰으로 변환
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True  # 커스텀 코드 실행 허용
        )
        
        # 모델 로드
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"✅ 비디오 AI 준비 완료! (디바이스: {self.device})")
    
    def extract_frames(self, video_path, num_frames=8):
        """비디오에서 균등하게 프레임을 추출하는 함수"""
        print(f"🎞️ 비디오에서 {num_frames}개의 프레임을 추출합니다...")
        
        # 비디오 읽기
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # 균등한 간격으로 프레임 선택
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame)
            frames.append(frame_pil)
        
        return frames, indices, total_frames
    
    def analyze_video(self, video_path, question="이 비디오에서 무슨 일이 일어나고 있나요?"):
        """비디오를 분석하고 질문에 답하는 함수"""
        
        # 1. 프레임 추출
        frames, indices, total_frames = self.extract_frames(video_path)
        
        # 2. 프레임 시각화
        self.visualize_frames(frames, indices, total_frames)
        
        # 3. 모델에 입력
        print(f"\n🤔 AI가 비디오를 분석하고 있습니다...")
        print(f"질문: {question}")
        
        # 입력 준비
        msgs = [
            {
                'role': 'user',
                'content': [
                    *[{'type': 'image', 'image': frame} for frame in frames],
                    {'type': 'text', 'text': question}
                ]
            }
        ]
        
        # 추론
        with torch.no_grad():
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                max_new_tokens=200,
                temperature=0.7
            )
        
        print(f"\n🤖 AI의 답변: {answer}")
        
        return answer, frames
    
    def visualize_frames(self, frames, indices, total_frames):
        """추출된 프레임을 시각화하는 함수"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, (frame, idx) in enumerate(zip(frames, indices)):
            axes[i].imshow(frame)
            axes[i].set_title(f'프레임 {idx}/{total_frames}')
            axes[i].axis('off')
        
        plt.suptitle('비디오에서 추출한 프레임들', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

# 사용 예시 (실제 비디오 파일이 필요합니다)
video_ai = VideoUnderstandingAI()

# 비디오 분석 예제
video_questions = [
    "이 비디오에서 무슨 일이 일어나고 있나요?",
    "주요 등장인물은 누구인가요?",
    "어떤 감정이 느껴지나요?",
    "이 비디오의 주제는 무엇인가요?"
]

# 여러 질문으로 분석
for question in video_questions:
    answer, frames = video_ai.analyze_video("sample/dog.mp4", question)
