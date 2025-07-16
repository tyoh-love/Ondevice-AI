# practical_optimization.py - 실제 모델 최적화

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import platform

# YOLO 모델 관련 import
try:
    from yolo_models import YOLOModelLoader, YOLOOptimizer
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class OptimizableModel(nn.Module):
    """최적화할 예시 모델"""
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ModelOptimizer:
    """모델 최적화 도구"""
    
    def __init__(self, explain_mode=False, model_type='simple'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 디바이스: {self.device}")
        
        # 플랫폼 체크
        self.platform = platform.system()
        self.is_arm = platform.machine().startswith('arm') or platform.machine() == 'aarch64'
        
        if self.platform == 'Darwin' or self.is_arm:
            print(f"⚠️  플랫폼: {self.platform} ({platform.machine()}) - 양자화가 제한적일 수 있습니다.")
        
        # 한글 폰트 설정
        self.setup_korean_font()
        
        # 설명 모드
        self.explain_mode = explain_mode
        self.model_type = model_type
        
        if self.explain_mode:
            print("\n📚 교육 모드가 활성화되었습니다. 각 단계를 자세히 설명합니다.\n")
        
        # YOLO 모델 관련 초기화
        if model_type == 'yolo' and YOLO_AVAILABLE:
            self.yolo_loader = YOLOModelLoader(explain_mode=explain_mode)
            self.yolo_optimizer = YOLOOptimizer(explain_mode=explain_mode)
            if explain_mode:
                print("🎯 YOLO 모델 모드가 활성화되었습니다.")
        elif model_type == 'yolo' and not YOLO_AVAILABLE:
            print("⚠️  YOLO 모델을 사용하려면 ultralytics 패키지가 필요합니다.")
            print("   설치 방법: pip install ultralytics")
            self.model_type = 'simple'
    
    def setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            if self.platform == 'Darwin':  # macOS
                font_paths = [
                    '/Library/Fonts/AppleGothic.ttf',
                    '/System/Library/Fonts/AppleSDGothicNeo.ttc',
                    '/Library/Fonts/NanumGothic.ttf',
                    '/Library/Fonts/NanumBarunGothic.ttf'
                ]
                
                font_found = False
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font_prop = fm.FontProperties(fname=font_path)
                        plt.rcParams['font.family'] = font_prop.get_name()
                        plt.rcParams['axes.unicode_minus'] = False
                        font_found = True
                        print(f"✅ 한글 폰트 설정: {font_prop.get_name()}")
                        break
                
                if not font_found:
                    print("⚠️  한글 폰트를 찾을 수 없습니다. 차트의 한글이 깨질 수 있습니다.")
                    plt.rcParams['font.family'] = 'DejaVu Sans'
                    
            elif self.platform == 'Windows':
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
            else:  # Linux
                plt.rcParams['font.family'] = 'NanumGothic'
                plt.rcParams['axes.unicode_minus'] = False
                
        except Exception as e:
            print(f"⚠️  폰트 설정 중 오류: {e}")
            # Fallback to English labels
            self.use_english_labels = True
        else:
            self.use_english_labels = False
    
    def print_ascii_art(self, art_type):
        """ASCII 아트로 최적화 방법 시각화"""
        if art_type == "quantization":
            print("🎨 양자화 과정 시각화:")
            print("    32-bit 가중치     →     8-bit 가중치")
            print("    [0.123456789]     →     [123]")
            print("    [0.987654321]     →     [246]")
            print("    [0.555555555]     →     [138]")
            print("    ┌─────────────┐   →   ┌─────────┐")
            print("    │  4 bytes    │   →   │ 1 byte  │")
            print("    └─────────────┘   →   └─────────┘")
            print("    💾 메모리 사용량: 75% 감소!")
        
        elif art_type == "pruning":
            print("🎨 가지치기 과정 시각화:")
            print("    원본 신경망           →     가지치기된 신경망")
            print("    ●───●───●───●         →     ●───●───●───●")
            print("    │╲ ╱│╲ ╱│╲ ╱│         →     │   │   │   │")
            print("    │ ╳ │ ╳ │ ╳ │         →     │   │   │   │")
            print("    │╱ ╲│╱ ╲│╱ ╲│         →     │   │   │   │")
            print("    ●───●───●───●         →     ●───●───●───●")
            print("    🔗 연결: 16개          →     🔗 연결: 8개")
            print("    ✂️ 중요하지 않은 연결 제거!")
        
        elif art_type == "distillation":
            print("🎨 지식 증류 과정 시각화:")
            print("    교사 모델 (큰 모델)     →     학생 모델 (작은 모델)")
            print("    ●───●───●───●───●       →     ●───●───●")
            print("    │╲ ╱│╲ ╱│╲ ╱│╲ ╱│       →     │╲ ╱│╲ ╱│")
            print("    │ ╳ │ ╳ │ ╳ │ ╳ │       →     │ ╳ │ ╳ │")
            print("    │╱ ╲│╱ ╲│╱ ╲│╱ ╲│       →     │╱ ╲│╱ ╲│")
            print("    ●───●───●───●───●       →     ●───●───●")
            print("    🧠 지식 전달: 큰 모델 → 작은 모델")
            print("    📚 성능 유지하면서 크기 감소!")
        
        print()
    
    def explain_concept(self, concept):
        """개념 설명"""
        if not self.explain_mode:
            return
            
        explanations = {
            "quantization": """
📖 양자화란?
• 정의: 32비트 부동소수점 → 8비트 정수로 변환
• 비유: 고화질 사진을 압축하여 용량 줄이기
• 장점: 메모리 사용량 4배 감소, 추론 속도 향상
• 단점: 정확도 약간 감소 가능
""",
            "pruning": """
📖 가지치기란?
• 정의: 중요하지 않은 신경망 연결 제거
• 비유: 나무 가지치기 - 불필요한 가지 제거
• 장점: 모델 크기 감소, 추론 속도 향상
• 단점: 과도한 가지치기 시 성능 저하
""",
            "distillation": """
📖 지식 증류란?
• 정의: 큰 모델(교사)의 지식을 작은 모델(학생)에게 전달
• 비유: 선생님이 학생에게 핵심 지식 전수
• 장점: 작은 모델로 큰 모델 성능 근사
• 단점: 추가 훈련 시간 필요
"""
        }
        
        if concept in explanations:
            print(explanations[concept])
    
    def visualize_model_structure(self, model, name="모델"):
        """모델 구조 시각화"""
        print(f"\n🏗️ {name} 구조:")
        print("=" * 50)
        
        total_params = 0
        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            print(f"📊 {name:15} | 형태: {str(param.shape):20} | 파라미터: {param_count:,}")
        
        print("=" * 50)
        print(f"🔢 총 파라미터 수: {total_params:,}")
        
        # 메모리 사용량 계산
        memory_mb = total_params * 4 / (1024 * 1024)  # 32-bit float
        print(f"💾 메모리 사용량: {memory_mb:.2f} MB")
        
        if self.explain_mode:
            print("\n💡 구조 설명:")
            print("• fc1: 입력층 → 첫 번째 은닉층 (784 → 256/64)")
            print("• fc2: 첫 번째 은닉층 → 두 번째 은닉층 (256/64 → 256/64)")
            print("• fc3: 두 번째 은닉층 → 출력층 (256/64 → 10)")
            print("• ReLU: 비선형 활성화 함수")
        
        print()
    
    def plot_weight_histogram(self, model, name="모델"):
        """가중치 히스토그램 시각화"""
        if not self.explain_mode:
            return
            
        print(f"\n📊 {name} 가중치 분포 분석:")
        
        # 모든 가중치를 수집
        all_weights = []
        layer_weights = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                layer_weights[name] = weights
        
        # 기본 통계 출력
        import numpy as np
        all_weights = np.array(all_weights)
        
        print(f"📈 전체 가중치 통계:")
        print(f"  • 평균: {np.mean(all_weights):.4f}")
        print(f"  • 표준편차: {np.std(all_weights):.4f}")
        print(f"  • 최솟값: {np.min(all_weights):.4f}")
        print(f"  • 최댓값: {np.max(all_weights):.4f}")
        print(f"  • 0에 가까운 값 (절댓값 < 0.01): {np.sum(np.abs(all_weights) < 0.01):,}개 ({np.sum(np.abs(all_weights) < 0.01)/len(all_weights)*100:.1f}%)")
        
        # 레이어별 통계
        print(f"\n📊 레이어별 가중치 분포:")
        for layer_name, weights in layer_weights.items():
            print(f"  • {layer_name}: 평균={np.mean(weights):.4f}, 표준편차={np.std(weights):.4f}")
        
        # 시각화 생성 (matplotlib가 있는 경우)
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'{name} 가중치 분포', fontsize=16)
            
            # 전체 가중치 히스토그램
            axes[0, 0].hist(all_weights, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('전체 가중치 분포')
            axes[0, 0].set_xlabel('가중치 값')
            axes[0, 0].set_ylabel('빈도')
            axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
            
            # 0 근처 확대
            small_weights = all_weights[np.abs(all_weights) < 0.1]
            axes[0, 1].hist(small_weights, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('0 근처 가중치 분포 (절댓값 < 0.1)')
            axes[0, 1].set_xlabel('가중치 값')
            axes[0, 1].set_ylabel('빈도')
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            
            # 레이어별 비교 (처음 두 레이어)
            layer_names = list(layer_weights.keys())[:2]
            for i, layer_name in enumerate(layer_names):
                weights = layer_weights[layer_name]
                axes[1, i].hist(weights, bins=30, alpha=0.7, color=['orange', 'purple'][i])
                axes[1, i].set_title(f'{layer_name} 가중치 분포')
                axes[1, i].set_xlabel('가중치 값')
                axes[1, i].set_ylabel('빈도')
                axes[1, i].axvline(0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"  ⚠️ 히스토그램 시각화 오류: {e}")
        
        print()
    
    def create_dummy_data(self, num_samples=1000, device=None):
        """테스트용 더미 데이터 생성"""
        if device is None:
            device = 'cpu'  # Default to CPU to avoid issues with quantization
        X = torch.randn(num_samples, 784, device=device)
        y = torch.randint(0, 10, (num_samples,), device=device)
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    def measure_performance(self, model, dataloader, name="Model", model_type=None):
        """모델 성능 측정"""
        print(f"\n⏱️  {name} 성능 측정 중...")
        model.eval()
        
        # 크기 측정 - YOLO 모델과 일반 모델 구분
        print("  - 모델 크기 계산 중...")
        
        # 모델 타입 자동 감지
        if model_type is None:
            model_type = self.model_type
        
        if model_type == 'yolo' and hasattr(model, 'model'):
            # YOLO 모델의 경우 파라미터 수 기반 계산
            pytorch_model = model.model
            total_params = sum(p.numel() for p in pytorch_model.parameters())
            model_size = total_params * 4 / (1024 * 1024)  # 32-bit float assumption
            print(f"  - YOLO 모델 ({total_params:,} 파라미터)")
        else:
            # 일반 모델의 경우 state_dict 저장 방식
            torch.save(model.state_dict(), 'temp_model.pth')
            model_size = os.path.getsize('temp_model.pth') / (1024 * 1024)  # MB
            os.remove('temp_model.pth')
            print(f"  - 일반 모델 (state_dict 기반)")
        
        # 속도 측정
        print("  - 추론 속도 측정 중...")
        total_time = 0
        num_batches = 0
        
        with torch.no_grad():
            for X, _ in dataloader:
                # Move input to the same device as the model
                # More robust quantized model detection
                is_quantized = (
                    'quantized' in str(type(model)).lower() or 
                    hasattr(model, '_is_quantized') or
                    any('quantized' in str(type(m)).lower() for m in model.modules()) or
                    any(hasattr(m, '_packed_params') for m in model.modules())  # Common in quantized layers
                )
                
                if is_quantized:
                    # Quantized models MUST run on CPU
                    model_device = torch.device('cpu')
                    # Ensure model is on CPU
                    if hasattr(model, 'to'):
                        model = model.cpu()
                else:
                    try:
                        model_device = next(model.parameters()).device
                    except StopIteration:
                        # If model has no parameters, default to CPU for safety
                        model_device = torch.device('cpu')
                
                # Move input to model device
                X = X.to(model_device)
                
                start_time = time.time()
                _ = model(X)
                
                # GPU 동기화 (정확한 시간 측정)
                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                
                total_time += time.time() - start_time
                num_batches += 1
                
                if num_batches >= 10:  # 10배치만 테스트
                    break
        
        avg_inference_time = total_time / num_batches * 1000  # ms
        
        print(f"\n{name} 성능:")
        print(f"  - 모델 크기: {model_size:.2f} MB")
        print(f"  - 평균 추론 시간: {avg_inference_time:.2f} ms/batch")
        
        return model_size, avg_inference_time
    
    def quantize_dynamic(self, model):
        """동적 양자화 적용"""
        print("\n🔄 동적 양자화 적용 중...")
        
        # 교육 모드에서 설명 및 시각화
        if self.explain_mode:
            self.explain_concept("quantization")
            self.print_ascii_art("quantization")
        
        try:
            print("  📋 1단계: 모델 분석 중...")
            if self.explain_mode:
                print("    • Linear 레이어를 찾아 양자화 대상 선정")
                print("    • 현재 가중치 형태: 32-bit 부동소수점")
            
            print("  🔧 2단계: 동적 양자화 적용...")
            if self.explain_mode:
                print("    • 런타임에 동적으로 가중치를 8-bit로 변환")
                print("    • 입력 데이터에 따라 스케일 조정")
            
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Linear 레이어만 양자화
                dtype=torch.qint8
            )
            
            # Ensure quantized model stays on CPU
            quantized_model = quantized_model.cpu()
            
            print("  ✅ 3단계: 양자화 완료!")
            if self.explain_mode:
                print("    • 가중치가 8-bit 정수로 변환됨")
                print("    • 메모리 사용량 약 4배 감소")
                print("    • 추론 속도 향상")
                print("    • 모델이 CPU에서 실행됨 (양자화 요구사항)")
            
            return quantized_model
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e):
                print("  ⚠️  양자화가 이 플랫폼에서 지원되지 않습니다 (macOS/ARM).")
                print("  ℹ️  대신 가지치기와 지식 증류를 사용해 보세요.")
                return None
            else:
                raise
    
    def quantize_static(self, model, dataloader):
        """정적 양자화 적용"""
        print("\n정적 양자화 적용 중...")
        
        try:
            # 모델 준비
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # 퓨전 가능한 모듈 결합
            model_fused = torch.quantization.fuse_modules(
                model, 
                [['fc1', 'relu1'], ['fc2', 'relu2']]
            )
            
            # 양자화 준비
            model_prepared = torch.quantization.prepare(model_fused)
            
            # 캘리브레이션
            print("  캘리브레이션 실행 중...")
            with torch.no_grad():
                for i, (X, _) in enumerate(dataloader):
                    model_prepared(X)
                    print(f"    - 배치 {i+1}/10 처리 중...", end='\r')
                    if i >= 10:  # 10배치로 캘리브레이션
                        break
            print("    - 캘리브레이션 완료!    ")
            
            # 양자화 변환
            model_quantized = torch.quantization.convert(model_prepared)
            
            return model_quantized
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized" in str(e):
                print("  ⚠️  정적 양자화가 이 플랫폼에서 지원되지 않습니다.")
                return None
            else:
                raise
    
    def prune_model(self, model, sparsity=0.5):
        """가지치기 적용"""
        print(f"\n✂️ 가지치기 적용 중... (희소성: {sparsity*100}%)")
        
        # 교육 모드에서 설명 및 시각화
        if self.explain_mode:
            self.explain_concept("pruning")
            self.print_ascii_art("pruning")
        
        import torch.nn.utils.prune as prune
        
        print("  📋 1단계: 가중치 중요도 분석...")
        if self.explain_mode:
            print("    • L1 norm을 사용해 가중치 중요도 계산")
            print("    • 작은 가중치들은 성능에 미치는 영향이 적음")
        
        # 각 Linear 레이어에 가지치기 적용
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_count += 1
                
        print(f"  ✂️ 2단계: {layer_count}개 레이어에 가지치기 적용...")
        current_layer = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                current_layer += 1
                if self.explain_mode:
                    print(f"    • {name} 레이어: {sparsity*100}%의 가중치 제거 중...")
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        print("  🔧 3단계: 가지치기 영구 적용...")
        if self.explain_mode:
            print("    • 마스크를 적용하여 가중치를 실제로 제거")
            print("    • 모델 구조는 유지하되 연결이 끊어짐")
        
        # 가지치기 영구 적용
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
        
        print("  ✅ 가지치기 완료!")
        if self.explain_mode:
            print(f"    • 전체 가중치의 {sparsity*100}%가 제거됨")
            print("    • 모델 크기 감소, 추론 속도 향상")
            self.plot_weight_histogram(model, f"가지치기 후 모델 ({sparsity*100}% 제거)")
        
        return model
    
    def knowledge_distillation(self, teacher_model, student_model, dataloader, epochs=5):
        """지식 증류"""
        print("\n🎓 지식 증류 수행 중...")
        
        # 교육 모드에서 설명 및 시각화
        if self.explain_mode:
            self.explain_concept("distillation")
            self.print_ascii_art("distillation")
        
        print("  📋 1단계: 학생 모델 준비...")
        # 학생 모델 (더 작은 모델)
        small_model = OptimizableModel(hidden_size=64)  # 더 작은 히든 크기
        small_model = small_model.to(self.device)
        
        if self.explain_mode:
            print("    • 교사 모델: 256개 히든 뉴런")
            print("    • 학생 모델: 64개 히든 뉴런 (4배 작음)")
            self.visualize_model_structure(teacher_model, "교사 모델")
            self.visualize_model_structure(small_model, "학생 모델")
        
        optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
        
        print("  🔧 2단계: 증류 손실 함수 설정...")
        # 증류 손실 함수
        def distillation_loss(student_logits, teacher_logits, temperature=3.0):
            soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
            soft_predictions = nn.functional.log_softmax(student_logits / temperature, dim=1)
            return nn.functional.kl_div(soft_predictions, soft_targets, reduction='batchmean') * temperature * temperature
        
        if self.explain_mode:
            print("    • KL Divergence 손실 사용")
            print("    • Temperature scaling으로 soft targets 생성")
        
        teacher_model.eval()
        
        print(f"  🎯 3단계: {epochs}번의 에포크로 학습...")
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            print(f"\n  📚 Epoch {epoch+1}/{epochs}:")
            if self.explain_mode:
                print("    • 교사 모델의 지식을 학생 모델에게 전수")
            
            for X, y in dataloader:
                X = X.to(self.device)
                
                # 교사 모델 예측
                with torch.no_grad():
                    teacher_logits = teacher_model(X)
                
                # 학생 모델 예측
                student_logits = small_model(X)
                
                # 손실 계산
                loss = distillation_loss(student_logits, teacher_logits)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                if batch_count % 5 == 0:
                    print(f"    - 배치 {batch_count}/{len(dataloader)} 처리 중...", end='\r')
            
            avg_loss = total_loss/len(dataloader)
            print(f"    - 평균 손실: {avg_loss:.4f}                    ")
            
            if self.explain_mode and epoch == 0:
                print("    • 손실이 감소할수록 학생이 교사를 더 잘 모방")
        
        print("  ✅ 지식 증류 완료!")
        if self.explain_mode:
            print("    • 작은 모델이 큰 모델의 지식을 성공적으로 학습")
            print("    • 크기는 작지만 비슷한 성능 달성")
        
        return small_model
    
    def save_model(self, model, filepath, model_type="optimized"):
        """최적화된 모델 저장"""
        print(f"\n💾 {model_type} 모델을 {filepath}에 저장 중...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'saved_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }, filepath)
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   ✅ 저장 완료! (크기: {file_size:.2f} MB)")
        return file_size
    
    def load_model(self, filepath, model_class=None):
        """저장된 모델 불러오기"""
        print(f"\n📂 {filepath}에서 모델 불러오는 중...")
        checkpoint = torch.load(filepath)
        
        if model_class is None:
            model_class = OptimizableModel
        
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   ✅ 모델 타입: {checkpoint.get('model_type', 'unknown')}")
        print(f"   ✅ 저장 시간: {checkpoint.get('saved_time', 'unknown')}")
        
        return model
    
    def compile_model(self, model):
        """모델 컴파일 (PyTorch 2.0+)"""
        if hasattr(torch, 'compile'):
            print("\n모델 컴파일 중...")
            compiled_model = torch.compile(model, mode="reduce-overhead")
            return compiled_model
        else:
            print("\nPyTorch 2.0 이상이 필요합니다.")
            return model
    
    def compare_all_optimizations(self):
        """모든 최적화 기법 비교"""
        print("\n=== 모델 최적화 종합 비교 ===\n")
        
        # 데이터 준비
        dataloader = self.create_dummy_data()
        
        # 원본 모델
        original_model = OptimizableModel().to(self.device)
        original_size, original_time = self.measure_performance(
            original_model, dataloader, "원본 모델"
        )
        
        results = {
            'Original': {'size': original_size, 'time': original_time}
        }
        
        # 1. 동적 양자화
        if self.device.type == 'cpu':
            dynamic_quantized = self.quantize_dynamic(original_model.cpu())
            if dynamic_quantized is not None:
                size, time = self.measure_performance(
                    dynamic_quantized, dataloader, "동적 양자화"
                )
                results['Dynamic Quantization'] = {'size': size, 'time': time}
            else:
                print("  ⏭️  동적 양자화를 건너뜁니다.")
        
        # 2. 가지치기
        import copy
        pruned_model = self.prune_model(copy.deepcopy(original_model), sparsity=0.5)
        size, time = self.measure_performance(
            pruned_model, dataloader, "가지치기 (50%)"
        )
        results['Pruning'] = {'size': size, 'time': time}
        
        # 3. 지식 증류
        student_model = self.knowledge_distillation(
            original_model, None, dataloader
        )
        size, time = self.measure_performance(
            student_model, dataloader, "지식 증류 (작은 모델)"
        )
        results['Knowledge Distillation'] = {'size': size, 'time': time}
        
        # 결과 시각화
        self.visualize_comparison(results)
        
        # 결과 표 출력
        self.print_summary_table(results)
        
        return results
    
    def visualize_comparison(self, results):
        """최적화 결과 비교 시각화"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = list(results.keys())
        sizes = [results[m]['size'] for m in methods]
        times = [results[m]['time'] for m in methods]
        
        # English labels if Korean font not available
        if hasattr(self, 'use_english_labels') and self.use_english_labels:
            size_label = 'Model Size (MB)'
            size_title = 'Model Size Comparison'
            time_label = 'Inference Time (ms/batch)'
            time_title = 'Inference Speed Comparison'
        else:
            size_label = '모델 크기 (MB)'
            size_title = '모델 크기 비교'
            time_label = '추론 시간 (ms/batch)'
            time_title = '추론 속도 비교'
        
        # 모델 크기 비교
        bars1 = ax1.bar(methods, sizes, color=['red', 'green', 'blue', 'orange'][:len(methods)])
        ax1.set_ylabel(size_label)
        ax1.set_title(size_title, fontweight='bold')
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # 크기 감소율 표시
        for i, bar in enumerate(bars1):
            if i > 0:
                reduction = (sizes[0] - sizes[i]) / sizes[0] * 100
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'-{reduction:.0f}%', ha='center', fontweight='bold')
        
        # 추론 시간 비교
        bars2 = ax2.bar(methods, times, color=['red', 'green', 'blue', 'orange'][:len(methods)])
        ax2.set_ylabel(time_label)
        ax2.set_title(time_title, fontweight='bold')
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        # 속도 향상 표시
        for i, bar in enumerate(bars2):
            if i > 0:
                speedup = times[0] / times[i]
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{speedup:.1f}x', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_table(self, results):
        """결과 요약 표 출력"""
        print("\n\n📊 === 최적화 결과 요약 표 === 📊")
        print("-" * 85)
        print(f"{'방법':^25} | {'모델 크기 (MB)':^20} | {'추론 시간 (ms)':^20} | {'크기 감소':^15}")
        print("-" * 85)
        
        original_size = results['Original']['size']
        original_time = results['Original']['time']
        
        for method, data in results.items():
            size = data['size']
            time = data['time']
            
            if method == 'Original':
                size_reduction = "-"
                time_speedup = "-"
            else:
                size_reduction = f"{(original_size - size) / original_size * 100:.1f}%"
                time_speedup = f"{original_time / time:.1f}x"
            
            print(f"{method:^25} | {size:^20.2f} | {time:^20.2f} | {size_reduction:^15}")
        
        print("-" * 85)
        
        # 최적 방법 찾기
        best_size_method = min(results.items(), key=lambda x: x[1]['size'])[0]
        best_time_method = min(results.items(), key=lambda x: x[1]['time'])[0]
        
        print(f"\n✨ 최적 결과:")
        print(f"  - 가장 작은 모델: {best_size_method} ({results[best_size_method]['size']:.2f} MB)")
        print(f"  - 가장 빠른 모델: {best_time_method} ({results[best_time_method]['time']:.2f} ms/batch)")
    
    def demo_dynamic_quantization(self):
        """동적 양자화 데모"""
        print("\n=== 동적 양자화 데모 ===\n")
        
        # 데이터 준비 - CPU용 데이터로더 생성 (양자화는 CPU에서만 작동)
        dataloader = self.create_dummy_data(device='cpu')
        
        # 원본 모델
        original_model = OptimizableModel().cpu()  # 동적 양자화는 CPU에서만 작동
        original_size, original_time = self.measure_performance(
            original_model, dataloader, "원본 모델"
        )
        
        # 동적 양자화 적용
        quantized_model = self.quantize_dynamic(original_model)
        if quantized_model is not None:
            quantized_size, quantized_time = self.measure_performance(
                quantized_model, dataloader, "동적 양자화 모델"
            )
            
            # 결과 요약
            print(f"\n📊 동적 양자화 결과:")
            print(f"  - 크기 감소: {original_size:.2f}MB → {quantized_size:.2f}MB ({(original_size-quantized_size)/original_size*100:.1f}% 감소)")
            print(f"  - 속도 향상: {original_time:.2f}ms → {quantized_time:.2f}ms ({original_time/quantized_time:.1f}x 빠름)")
            
            return quantized_model
        else:
            print("\n💡 양자화 대신 다른 최적화 기법을 시도해 보세요.")
            return None
    
    def demo_pruning(self, sparsity=0.5):
        """가지치기 데모"""
        print(f"\n=== 가지치기 데모 (희소성 {sparsity*100}%) ===\n")
        
        # 데이터 준비
        dataloader = self.create_dummy_data()
        
        # 원본 모델
        original_model = OptimizableModel().to(self.device)
        original_size, original_time = self.measure_performance(
            original_model, dataloader, "원본 모델"
        )
        
        # 가지치기 적용
        import copy
        pruned_model = self.prune_model(copy.deepcopy(original_model), sparsity=sparsity)
        pruned_size, pruned_time = self.measure_performance(
            pruned_model, dataloader, f"가지치기 모델 ({sparsity*100}%)"
        )
        
        # 결과 요약
        print(f"\n📊 가지치기 결과:")
        print(f"  - 크기 감소: {original_size:.2f}MB → {pruned_size:.2f}MB ({(original_size-pruned_size)/original_size*100:.1f}% 감소)")
        print(f"  - 속도 향상: {original_time:.2f}ms → {pruned_time:.2f}ms ({original_time/pruned_time:.1f}x 빠름)")
        
        return pruned_model
    
    def demo_knowledge_distillation(self):
        """지식 증류 데모"""
        print("\n=== 지식 증류 데모 ===\n")
        
        # 데이터 준비
        dataloader = self.create_dummy_data()
        
        # 교사 모델 (큰 모델)
        teacher_model = OptimizableModel(hidden_size=256).to(self.device)
        teacher_size, teacher_time = self.measure_performance(
            teacher_model, dataloader, "교사 모델 (큰 모델)"
        )
        
        # 지식 증류로 학생 모델 훈련
        student_model = self.knowledge_distillation(teacher_model, None, dataloader)
        student_size, student_time = self.measure_performance(
            student_model, dataloader, "학생 모델 (작은 모델)"
        )
        
        # 결과 요약
        print(f"\n📊 지식 증류 결과:")
        print(f"  - 크기 감소: {teacher_size:.2f}MB → {student_size:.2f}MB ({(teacher_size-student_size)/teacher_size*100:.1f}% 감소)")
        print(f"  - 속도 향상: {teacher_time:.2f}ms → {student_time:.2f}ms ({teacher_time/student_time:.1f}x 빠름)")
        
        return student_model
    
    def demo_yolo_pruning(self, model_name='yolov8s', sparsity=0.5):
        """YOLO 모델 가지치기 데모"""
        if self.model_type != 'yolo':
            print("⚠️  YOLO 모델 모드가 아닙니다.")
            return None
        
        print(f"\n=== YOLO 가지치기 데모 ({model_name}) ===\n")
        
        # 원본 모델 로드
        original_model = self.yolo_loader.load_model(model_name)
        original_info = self.yolo_loader.get_model_info(original_model)
        
        # 샘플 이미지 준비
        images = self.yolo_loader.create_sample_images(5)
        
        # 원본 모델 성능 측정
        print("📊 원본 모델 성능 측정...")
        original_benchmark = self.yolo_loader.benchmark_inference(original_model, images, runs=5)
        
        # 가지치기 적용
        pruned_model = self.yolo_optimizer.prune_yolo_model(original_model, sparsity=sparsity)
        pruned_info = self.yolo_loader.get_model_info(pruned_model)
        
        # 가지치기된 모델 성능 측정
        print("📊 가지치기된 모델 성능 측정...")
        pruned_benchmark = self.yolo_loader.benchmark_inference(pruned_model, images, runs=5)
        
        # 결과 비교
        print(f"\n📊 YOLO 가지치기 결과:")
        print(f"  - 모델 크기: {original_info['model_size_mb']:.2f}MB → {pruned_info['model_size_mb']:.2f}MB")
        size_reduction = (original_info['model_size_mb'] - pruned_info['model_size_mb']) / original_info['model_size_mb'] * 100
        print(f"  - 크기 감소: {size_reduction:.1f}%")
        print(f"  - FPS: {original_benchmark['fps']:.1f} → {pruned_benchmark['fps']:.1f}")
        fps_change = (pruned_benchmark['fps'] - original_benchmark['fps']) / original_benchmark['fps'] * 100
        print(f"  - FPS 변화: {fps_change:+.1f}%")
        
        return pruned_model
    
    def demo_yolo_distillation(self, teacher_model='yolov8m', student_model='yolov8n'):
        """YOLO 지식 증류 데모"""
        if self.model_type != 'yolo':
            print("⚠️  YOLO 모델 모드가 아닙니다.")
            return None
        
        print(f"\n=== YOLO 지식 증류 데모 ({teacher_model} → {student_model}) ===\n")
        
        # 교사 모델 로드
        teacher = self.yolo_loader.load_model(teacher_model)
        teacher_info = self.yolo_loader.get_model_info(teacher)
        
        # 학생 모델 로드
        student = self.yolo_loader.load_model(student_model)
        student_info = self.yolo_loader.get_model_info(student)
        
        # 샘플 이미지 준비
        images = self.yolo_loader.create_sample_images(5)
        
        # 교사 모델 성능 측정
        print("📊 교사 모델 성능 측정...")
        teacher_benchmark = self.yolo_loader.benchmark_inference(teacher, images, runs=5)
        
        # 학생 모델 성능 측정
        print("📊 학생 모델 성능 측정...")
        student_benchmark = self.yolo_loader.benchmark_inference(student, images, runs=5)
        
        # 결과 비교
        print(f"\n📊 YOLO 지식 증류 결과:")
        print(f"  - 교사 모델 크기: {teacher_info['model_size_mb']:.2f}MB")
        print(f"  - 학생 모델 크기: {student_info['model_size_mb']:.2f}MB")
        size_reduction = (teacher_info['model_size_mb'] - student_info['model_size_mb']) / teacher_info['model_size_mb'] * 100
        print(f"  - 크기 감소: {size_reduction:.1f}%")
        print(f"  - 교사 FPS: {teacher_benchmark['fps']:.1f}")
        print(f"  - 학생 FPS: {student_benchmark['fps']:.1f}")
        speedup = student_benchmark['fps'] / teacher_benchmark['fps']
        print(f"  - 속도 향상: {speedup:.1f}x")
        
        return student
    
    def demo_yolo_comprehensive(self):
        """YOLO 종합 최적화 데모"""
        if self.model_type != 'yolo':
            print("⚠️  YOLO 모델 모드가 아닙니다.")
            return None
        
        print("\n=== YOLO 종합 최적화 데모 ===\n")
        
        # 다양한 YOLO 모델 비교
        models_to_test = ['yolov8n', 'yolov8s', 'yolov8m']
        results = {}
        
        # 샘플 이미지 준비
        images = self.yolo_loader.create_sample_images(5)
        
        for model_name in models_to_test:
            print(f"\n🔄 {model_name} 모델 테스트 중...")
            
            # 모델 로드
            model = self.yolo_loader.load_model(model_name)
            model_info = self.yolo_loader.get_model_info(model)
            
            # 성능 측정
            benchmark = self.yolo_loader.benchmark_inference(model, images, runs=3)
            
            # 가지치기 테스트
            pruned_model = self.yolo_optimizer.prune_yolo_model(model, sparsity=0.3)
            pruned_benchmark = self.yolo_loader.benchmark_inference(pruned_model, images, runs=3)
            
            results[model_name] = {
                'original': {
                    'size_mb': model_info['model_size_mb'],
                    'fps': benchmark['fps'],
                    'params': model_info['total_params']
                },
                'pruned': {
                    'fps': pruned_benchmark['fps']
                }
            }
        
        # 결과 표 출력
        print(f"\n📊 === YOLO 모델 종합 비교 === 📊")
        print("-" * 80)
        print(f"{'모델':^12} | {'크기(MB)':^10} | {'파라미터':^12} | {'원본 FPS':^10} | {'가지치기 FPS':^12}")
        print("-" * 80)
        
        for model_name, data in results.items():
            print(f"{model_name:^12} | {data['original']['size_mb']:^10.1f} | {data['original']['params']:^12,} | {data['original']['fps']:^10.1f} | {data['pruned']['fps']:^12.1f}")
        
        print("-" * 80)
        
        return results
    
    def demo_yolo_m_before_after(self):
        """YOLOv8m 모델 최적화 전후 비교 데모"""
        if self.model_type != 'yolo':
            print("⚠️  YOLO 모델 모드가 아닙니다.")
            return None
        
        print("\n=== YOLOv8m 최적화 전후 비교 데모 ===\n")
        
        # 샘플 이미지 준비
        images = self.yolo_loader.create_sample_images(5)
        
        # 1. 원본 YOLOv8m 모델 로드 및 측정
        print("🔄 1단계: 원본 YOLOv8m 모델 로드 및 성능 측정\n")
        original_model = self.yolo_loader.load_model('yolov8m')
        original_info = self.yolo_loader.get_model_info(original_model)
        
        print("📊 원본 모델 성능 측정...")
        original_benchmark = self.yolo_loader.benchmark_inference(original_model, images, runs=5)
        
        # 2. 가지치기 적용
        print(f"\n{'='*60}")
        print("🔄 2단계: 가지치기 적용 (30% 희소성)\n")
        
        # 가지치기를 위해 새 모델 로드 (deepcopy 대신)
        pruned_model = self.yolo_loader.load_model('yolov8m')
        pruned_model = self.yolo_optimizer.prune_yolo_model(pruned_model, sparsity=0.3)
        pruned_info = self.yolo_loader.get_model_info(pruned_model)
        
        print("📊 가지치기된 모델 성능 측정...")
        pruned_benchmark = self.yolo_loader.benchmark_inference(pruned_model, images, runs=5)
        
        # 3. 지식 증류 적용 (YOLOv8m → YOLOv8n)
        print(f"\n{'='*60}")
        print("🔄 3단계: 지식 증류 적용 (YOLOv8m → YOLOv8n)\n")
        
        student_model = self.yolo_loader.load_model('yolov8n')
        student_info = self.yolo_loader.get_model_info(student_model)
        
        print("📊 학생 모델 성능 측정...")
        student_benchmark = self.yolo_loader.benchmark_inference(student_model, images, runs=5)
        
        # 4. 결과 비교 및 시각화
        print(f"\n{'='*60}")
        print("📊 === YOLOv8m 최적화 전후 비교 결과 === 📊\n")
        
        # 상세 결과 표
        print("-" * 100)
        print(f"{'최적화 방법':^20} | {'모델 크기(MB)':^15} | {'파라미터 수':^15} | {'FPS':^10} | {'크기 감소':^12} | {'속도 변화':^12}")
        print("-" * 100)
        
        # 원본 모델
        print(f"{'원본 (YOLOv8m)':^20} | {original_info['model_size_mb']:^15.1f} | {original_info['total_params']:^15,} | {original_benchmark['fps']:^10.1f} | {'-':^12} | {'-':^12}")
        
        # 가지치기 모델
        size_reduction_prune = (original_info['model_size_mb'] - pruned_info['model_size_mb']) / original_info['model_size_mb'] * 100
        fps_change_prune = (pruned_benchmark['fps'] - original_benchmark['fps']) / original_benchmark['fps'] * 100
        print(f"{'가지치기 (30%)':^20} | {pruned_info['model_size_mb']:^15.1f} | {pruned_info['total_params']:^15,} | {pruned_benchmark['fps']:^10.1f} | {size_reduction_prune:^12.1f}% | {fps_change_prune:^+12.1f}%")
        
        # 지식 증류 모델
        size_reduction_distill = (original_info['model_size_mb'] - student_info['model_size_mb']) / original_info['model_size_mb'] * 100
        fps_change_distill = (student_benchmark['fps'] - original_benchmark['fps']) / original_benchmark['fps'] * 100
        print(f"{'지식 증류 (→n)':^20} | {student_info['model_size_mb']:^15.1f} | {student_info['total_params']:^15,} | {student_benchmark['fps']:^10.1f} | {size_reduction_distill:^12.1f}% | {fps_change_distill:^+12.1f}%")
        
        print("-" * 100)
        
        # 핵심 인사이트
        print(f"\n🎯 === 핵심 인사이트 === 🎯")
        print(f"📦 모델 크기:")
        print(f"   • 원본: {original_info['model_size_mb']:.1f}MB")
        print(f"   • 가지치기: {pruned_info['model_size_mb']:.1f}MB ({size_reduction_prune:.1f}% 감소)")
        print(f"   • 지식 증류: {student_info['model_size_mb']:.1f}MB ({size_reduction_distill:.1f}% 감소)")
        
        print(f"\n⚡ 추론 속도:")
        print(f"   • 원본: {original_benchmark['fps']:.1f} FPS")
        print(f"   • 가지치기: {pruned_benchmark['fps']:.1f} FPS ({fps_change_prune:+.1f}% 변화)")
        print(f"   • 지식 증류: {student_benchmark['fps']:.1f} FPS ({fps_change_distill:+.1f}% 변화)")
        
        # 추천 방법
        best_size_method = "지식 증류" if size_reduction_distill > size_reduction_prune else "가지치기"
        best_speed_method = "지식 증류" if fps_change_distill > fps_change_prune else "가지치기"
        
        print(f"\n💡 === 추천 방법 === 💡")
        print(f"🏆 크기 감소 최고: {best_size_method}")
        print(f"🏆 속도 향상 최고: {best_speed_method}")
        
        if self.explain_mode:
            print(f"\n📚 === 교육적 설명 === 📚")
            print(f"🔍 가지치기 (Pruning):")
            print(f"   • 중요하지 않은 뉴런 연결을 제거")
            print(f"   • 모델 구조는 유지하되 가중치를 0으로 설정")
            print(f"   • 실제 크기 감소는 제한적이지만 연산량 감소")
            
            print(f"\n🎓 지식 증류 (Knowledge Distillation):")
            print(f"   • 큰 모델의 지식을 작은 모델로 전수")
            print(f"   • 드라마틱한 크기 감소 효과")
            print(f"   • 작은 모델이지만 큰 모델의 성능 근사")
        
        # 반환할 결과 데이터
        results = {
            'original': {
                'size_mb': original_info['model_size_mb'],
                'fps': original_benchmark['fps'],
                'params': original_info['total_params']
            },
            'pruned': {
                'size_mb': pruned_info['model_size_mb'],
                'fps': pruned_benchmark['fps'],
                'params': pruned_info['total_params'],
                'size_reduction': size_reduction_prune,
                'fps_change': fps_change_prune
            },
            'distilled': {
                'size_mb': student_info['model_size_mb'],
                'fps': student_benchmark['fps'],
                'params': student_info['total_params'],
                'size_reduction': size_reduction_distill,
                'fps_change': fps_change_distill
            }
        }
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='모델 최적화 데모')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'quantize', 'prune', 'distill', 'individual', 'yolo-prune', 'yolo-distill', 'yolo-comprehensive', 'yolo-m-demo'],
                        help='실행할 최적화 모드 선택')
    parser.add_argument('--model-type', type=str, default='simple',
                        choices=['simple', 'yolo'],
                        help='모델 타입 선택')
    parser.add_argument('--yolo-model', type=str, default='yolov8s',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLO 모델 선택')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='가지치기 희소성 (0-1)')
    parser.add_argument('--save-models', action='store_true',
                        help='최적화된 모델 저장')
    parser.add_argument('--no-plot', action='store_true',
                        help='그래프 표시 안 함')
    parser.add_argument('--explain', action='store_true',
                        help='교육 모드: 각 단계를 자세히 설명')
    
    args = parser.parse_args()
    
    # 최적화 실행
    optimizer = ModelOptimizer(explain_mode=args.explain, model_type=args.model_type)
    
    if args.mode == 'all':
        results = optimizer.compare_all_optimizations()
    elif args.mode == 'quantize':
        model = optimizer.demo_dynamic_quantization()
        if model is not None and args.save_models:
            optimizer.save_model(model, 'quantized_model.pth', 'quantized')
    elif args.mode == 'prune':
        model = optimizer.demo_pruning(sparsity=args.sparsity)
        if args.save_models:
            optimizer.save_model(model, f'pruned_model_{args.sparsity}.pth', f'pruned_{args.sparsity}')
    elif args.mode == 'distill':
        model = optimizer.demo_knowledge_distillation()
        if args.save_models:
            optimizer.save_model(model, 'distilled_model.pth', 'distilled')
    elif args.mode == 'individual':
        print("\n🚀 개별 최적화 데모 실행\n")
        print("1. 동적 양자화 데모")
        optimizer.demo_dynamic_quantization()
        print("\n" + "="*80 + "\n")
        
        print("2. 가지치기 데모")
        optimizer.demo_pruning()
        print("\n" + "="*80 + "\n")
        
        print("3. 지식 증류 데모")
        optimizer.demo_knowledge_distillation()
    elif args.mode == 'yolo-prune':
        optimizer.demo_yolo_pruning(model_name=args.yolo_model, sparsity=args.sparsity)
    elif args.mode == 'yolo-distill':
        optimizer.demo_yolo_distillation()
    elif args.mode == 'yolo-comprehensive':
        optimizer.demo_yolo_comprehensive()
    elif args.mode == 'yolo-m-demo':
        optimizer.demo_yolo_m_before_after()
