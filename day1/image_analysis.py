from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import matplotlib.pyplot as plt

class ImageUnderstandingAI:
    """이미지를 이해하는 똑똑한 AI"""
    
    def __init__(self):
        """MiniCPM-O 2.6 모델 초기화"""
        print("🖼️ 이미지 이해 AI를 준비하고 있습니다...")
        
        # MiniCPM-O-2.6 uses the same model as MiniCPM-V-2.6
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
        
        print(f"✅ 이미지 AI 준비 완료! (디바이스: {self.device})")
    
    def analyze_image(self, image_path, question="이 이미지에서 무엇이 보이나요?"):
        """이미지를 분석하고 질문에 답하는 함수"""
        
        # 1. 이미지 로드
        print(f"📷 이미지를 로드하고 있습니다: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # 2. 이미지 시각화
        self.visualize_image(image, image_path)
        
        # 3. 모델에 입력
        print(f"\n🤔 AI가 이미지를 분석하고 있습니다...")
        print(f"질문: {question}")
        
        # 입력 준비
        msgs = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
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
        
        return answer, image
    
    def visualize_image(self, image, image_path):
        """이미지를 시각화하는 함수"""
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f'분석할 이미지: {image_path}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# 메인 실행 코드
if __name__ == "__main__":
    # AI 초기화
    image_ai = ImageUnderstandingAI()
    
    # cry.png 이미지 분석
    image_path = "../cry.png"  # day1 폴더에서 실행시 상위 폴더의 cry.png
    
    # 여러 질문으로 분석
    questions = [
        "이 이미지에서 무엇이 보이나요?",
        "이미지의 분위기나 감정은 어떤가요?",
        "이미지에 있는 사람이나 캐릭터의 표정을 설명해주세요.",
        "이 이미지가 전달하려는 메시지는 무엇일까요?"
    ]
    
    print("\n" + "="*50)
    print("cry.png 이미지 분석 시작")
    print("="*50 + "\n")
    
    for i, question in enumerate(questions, 1):
        print(f"\n[질문 {i}]")
        answer, image = image_ai.analyze_image(image_path, question)
        print("-"*50)