# yolo_models.py - YOLO 모델 관리 및 최적화

import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
import os
from pathlib import Path

class YOLOModelLoader:
    """YOLO 모델 로딩 및 관리 클래스"""
    
    def __init__(self, explain_mode=False):
        self.explain_mode = explain_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.available_models = {
            'yolov8n': 'yolov8n.pt',  # nano - 가장 작음
            'yolov8s': 'yolov8s.pt',  # small
            'yolov8m': 'yolov8m.pt',  # medium
            'yolov8l': 'yolov8l.pt',  # large
            'yolov8x': 'yolov8x.pt'   # extra large - 가장 큼
        }
        
        if self.explain_mode:
            print("📋 YOLO 모델 정보:")
            print("  • YOLOv8n: 3.2M 파라미터, 6.2MB")
            print("  • YOLOv8s: 11.2M 파라미터, 21.4MB")
            print("  • YOLOv8m: 25.9M 파라미터, 49.7MB")
            print("  • YOLOv8l: 43.7M 파라미터, 83.7MB")
            print("  • YOLOv8x: 68.2M 파라미터, 130.5MB")
    
    def load_model(self, model_name='yolov8n', load_state_dict=True):
        """YOLO 모델 로딩"""
        if model_name not in self.available_models:
            raise ValueError(f"지원되지 않는 모델: {model_name}")
        
        model_path = self.available_models[model_name]
        
        if self.explain_mode:
            print(f"\n🔄 {model_name} 모델 로딩 중...")
            print(f"  • 모델 파일: {model_path}")
            print(f"  • 디바이스: {self.device}")
        
        # YOLO 모델 로드
        model = YOLO(model_path)
        
        if self.explain_mode:
            print(f"  ✅ {model_name} 모델 로딩 완료!")
        
        return model
    
    def get_model_info(self, model):
        """모델 정보 추출"""
        # PyTorch 모델 추출
        pytorch_model = model.model
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        
        # 모델 크기 계산 (MB)
        model_size_mb = total_params * 4 / (1024 * 1024)  # 32-bit float
        
        info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'device': next(pytorch_model.parameters()).device
        }
        
        if self.explain_mode:
            print(f"\n📊 모델 정보:")
            print(f"  • 총 파라미터: {total_params:,}")
            print(f"  • 훈련 가능한 파라미터: {trainable_params:,}")
            print(f"  • 모델 크기: {model_size_mb:.2f} MB")
            print(f"  • 디바이스: {info['device']}")
        
        return info
    
    def create_sample_images(self, num_images=10):
        """샘플 이미지 생성 (COCO 데이터셋에서 다운로드)"""
        sample_urls = [
            "https://ultralytics.com/images/bus.jpg",
            "https://ultralytics.com/images/zidane.jpg",
            "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=640",
            "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=640",
            "https://images.unsplash.com/photo-1552053831-71594a27632d?w=640"
        ]
        
        # 샘플 이미지 디렉터리 생성
        sample_dir = Path("sample_images")
        sample_dir.mkdir(exist_ok=True)
        
        images = []
        for i, url in enumerate(sample_urls[:num_images]):
            try:
                img_path = sample_dir / f"sample_{i+1}.jpg"
                
                if not img_path.exists():
                    if self.explain_mode:
                        print(f"  📥 샘플 이미지 {i+1} 다운로드 중...")
                    
                    response = requests.get(url, stream=True)
                    with open(img_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                # 이미지 로드
                img = Image.open(img_path)
                images.append(str(img_path))
                
            except Exception as e:
                if self.explain_mode:
                    print(f"  ⚠️ 이미지 {i+1} 로드 실패: {e}")
                continue
        
        if self.explain_mode:
            print(f"  ✅ {len(images)}개 샘플 이미지 준비 완료!")
        
        return images
    
    def benchmark_inference(self, model, images, runs=10):
        """추론 벤치마크"""
        if self.explain_mode:
            print(f"\n⏱️ 추론 성능 벤치마킹 ({runs}회 실행)...")
        
        # 워밍업
        for _ in range(3):
            model(images[0], verbose=False)
        
        # 벤치마크 실행
        import time
        times = []
        
        for i in range(runs):
            start_time = time.time()
            
            for img_path in images:
                results = model(img_path, verbose=False)
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            if self.explain_mode and (i + 1) % 3 == 0:
                print(f"    - 실행 {i+1}/{runs} 완료...")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = len(images) / avg_time
        
        benchmark_results = {
            'avg_time_per_batch': avg_time,
            'std_time': std_time,
            'fps': fps,
            'images_per_batch': len(images)
        }
        
        if self.explain_mode:
            print(f"\n📊 벤치마크 결과:")
            print(f"  • 평균 배치 처리 시간: {avg_time:.3f}초")
            print(f"  • 표준편차: {std_time:.3f}초")
            print(f"  • FPS: {fps:.1f}")
            print(f"  • 이미지 수: {len(images)}")
        
        return benchmark_results

class YOLOOptimizer:
    """YOLO 모델 최적화 클래스"""
    
    def __init__(self, explain_mode=False):
        self.explain_mode = explain_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prune_yolo_model(self, model, sparsity=0.5):
        """YOLO 모델 가지치기"""
        if self.explain_mode:
            print(f"\n✂️ YOLO 모델 가지치기 (희소성: {sparsity*100}%)")
            print("  📋 YOLO 모델은 컨볼루션 레이어 중심 구조")
            print("  📋 구조적 가지치기로 채널 단위 제거")
        
        pytorch_model = model.model
        
        # 컨볼루션 레이어 찾기
        conv_layers = []
        for name, module in pytorch_model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if self.explain_mode:
            print(f"  🔍 발견된 Conv2d 레이어: {len(conv_layers)}개")
        
        # 가지치기 적용 (간단한 magnitude-based pruning)
        import torch.nn.utils.prune as prune
        
        pruned_layers = 0
        for name, module in conv_layers:
            if module.weight.numel() > 100:  # 작은 레이어는 건드리지 않음
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                pruned_layers += 1
                
                if self.explain_mode and pruned_layers <= 5:
                    print(f"    • {name}: {module.weight.shape} 가지치기 적용")
        
        # 가지치기 영구 적용
        for name, module in conv_layers:
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
        
        if self.explain_mode:
            print(f"  ✅ {pruned_layers}개 레이어에 가지치기 완료!")
        
        return model
    
    def knowledge_distillation_yolo(self, teacher_model, student_model_name='yolov8n', images=None, epochs=5):
        """YOLO 지식 증류"""
        if self.explain_mode:
            print(f"\n🎓 YOLO 지식 증류")
            print(f"  • 교사 모델: 큰 YOLO 모델")
            print(f"  • 학생 모델: {student_model_name}")
        
        # 학생 모델 로드
        loader = YOLOModelLoader(explain_mode=self.explain_mode)
        student_model = loader.load_model(student_model_name)
        
        if self.explain_mode:
            print(f"  📚 지식 증류는 일반적으로 대규모 데이터셋 필요")
            print(f"  📚 현재는 모델 크기 비교 시연")
        
        # 교사와 학생 모델 정보 비교
        teacher_info = loader.get_model_info(teacher_model)
        student_info = loader.get_model_info(student_model)
        
        if self.explain_mode:
            print(f"\n📊 크기 비교:")
            print(f"  • 교사 모델: {teacher_info['model_size_mb']:.2f} MB")
            print(f"  • 학생 모델: {student_info['model_size_mb']:.2f} MB")
            reduction = (teacher_info['model_size_mb'] - student_info['model_size_mb']) / teacher_info['model_size_mb'] * 100
            print(f"  • 크기 감소: {reduction:.1f}%")
        
        return student_model

if __name__ == "__main__":
    # 테스트 코드
    loader = YOLOModelLoader(explain_mode=True)
    
    # 모델 로드
    model = loader.load_model('yolov8n')
    
    # 모델 정보 출력
    info = loader.get_model_info(model)
    
    # 샘플 이미지 생성
    images = loader.create_sample_images(3)
    
    # 벤치마크 실행
    results = loader.benchmark_inference(model, images, runs=3)