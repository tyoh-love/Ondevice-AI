# YOLO 모델 최적화 데모

실제 YOLO 모델을 사용한 현실적인 성능 비교 데모입니다.

## 🚀 주요 기능

### 1. YOLO 모델 지원
- **YOLOv8n**: 3.2M 파라미터, 6.2MB (가장 작고 빠름)
- **YOLOv8s**: 11.2M 파라미터, 21.4MB 
- **YOLOv8m**: 25.9M 파라미터, 49.7MB
- **YOLOv8l**: 43.7M 파라미터, 83.7MB
- **YOLOv8x**: 68.2M 파라미터, 130.5MB (가장 크고 정확)

### 2. 실제 이미지 처리
- 자동 샘플 이미지 다운로드
- 실제 객체 검출 추론
- FPS 성능 측정

### 3. 최적화 기법
- **가지치기 (Pruning)**: 컨볼루션 레이어 가중치 제거
- **지식 증류 (Knowledge Distillation)**: 큰 모델 → 작은 모델 지식 전수
- **종합 비교**: 여러 모델 동시 성능 비교

## 📊 실행 예시

### 기본 YOLO 가지치기 데모
```bash
python day3/quantize.py --model-type yolo --mode yolo-prune --yolo-model yolov8n --sparsity 0.3 --explain
```

### YOLO 지식 증류 데모  
```bash
python day3/quantize.py --model-type yolo --mode yolo-distill --explain
```

### 종합 성능 비교
```bash
python day3/quantize.py --model-type yolo --mode yolo-comprehensive --explain
```

## 📈 성능 결과 예시

### YOLOv8n 가지치기 결과
- **모델 크기**: 12.04MB → 12.02MB (0.2% 감소)
- **FPS**: 38.2 → 39.9 (4.4% 향상)
- **30% 가지치기 적용 시 성능 향상**

### YOLO 지식 증류 결과 (YOLOv8m → YOLOv8n)
- **크기 감소**: 98.8MB → 12.0MB (87.8% 감소)
- **속도 향상**: 15.1 FPS → 36.6 FPS (2.4배 빠름)
- **드라마틱한 경량화 효과**

### 종합 비교 결과
```
모델        크기(MB)    파라미터        원본 FPS    가지치기 FPS  
yolov8n     12.0       3,157,200       36.6        37.9    
yolov8s     42.6       11,166,560      28.1        28.4    
yolov8m     98.8       25,902,640      15.3        14.8    
```

## 🎯 실용적 장점

1. **현실적 성능**: 더미 데이터가 아닌 실제 이미지 처리
2. **실시간 메트릭**: FPS, 모델 크기, 파라미터 수 등
3. **배포 관점**: 실제 배포 시 고려사항 반영
4. **교육적 가치**: 실제 모델로 최적화 효과 체험

## 💡 활용 방안

- **모델 선택**: 용도에 맞는 YOLO 모델 크기 결정
- **최적화 효과**: 가지치기/지식증류의 실제 효과 확인
- **성능 트레이드오프**: 정확도 vs 속도 균형점 탐색
- **배포 전략**: 엣지 디바이스 배포를 위한 최적화

## 🔧 요구사항

- Python 3.7+
- PyTorch
- ultralytics (YOLO 패키지)
- matplotlib
- numpy
- PIL

## 🎨 교육 모드

`--explain` 플래그로 활성화:
- 각 단계별 상세 설명
- ASCII 아트로 시각화
- 실시간 진행 상황 표시
- 모델 구조 분석
- 성능 개선 이유 설명