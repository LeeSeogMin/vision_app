# Week 10: 자율주행 인식 시스템 강의 자료

## 목차
1. [자율주행 이론](#1-자율주행-이론)
2. [차선 인식](#2-차선-인식)
3. [객체 탐지 및 추적](#3-객체-탐지-및-추적)
4. [통합 파이프라인](#4-통합-파이프라인)
5. [실전 배포](#5-실전-배포)

---

## 1. 자율주행 이론

### 1.1 SAE 자율주행 레벨

| 레벨 | 이름 | 설명 | 예시 |
|-----|------|------|------|
| 0 | 완전 수동 | 운전자가 모든 제어 | 일반 차량 |
| 1 | 운전자 보조 | 단일 기능 지원 | ACC, LKA |
| 2 | 부분 자동화 | 조향+가감속 동시 | Tesla Autopilot |
| 3 | 조건부 자동화 | 특정 조건 자율, 개입 가능 | Audi A8 Traffic Jam Pilot |
| 4 | 고도 자동화 | 대부분 자율 | Waymo One |
| 5 | 완전 자동화 | 모든 상황 자율 | 미래 기술 |

### 1.2 자율주행 시스템 구조

```
┌─────────────────────────────────────────────┐
│           센서 (Sensors)                    │
│  카메라 | 라이다 | 레이더 | 초음파 | GPS    │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│        1단계: 인식 (Perception)              │
│  • 차선 인식                                 │
│  • 객체 탐지 (차량, 보행자, 신호등)          │
│  • 객체 추적 (ID 유지)                       │
│  • 거리 추정                                 │
│  • 세그멘테이션                              │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│        2단계: 판단 (Planning)                │
│  • Mission Planning (경로 계획)             │
│  • Behavioral Planning (행동 결정)          │
│  • Motion Planning (궤적 생성)              │
│  • 위험도 분석                               │
└──────────────┬──────────────────────────────┘
               ↓
┌─────────────────────────────────────────────┐
│        3단계: 제어 (Control)                 │
│  • 조향 제어 (Steering)                      │
│  • 가속/감속 (Throttle/Brake)                │
│  • PID, MPC 알고리즘                         │
└─────────────────────────────────────────────┘
```

---

## 2. 차선 인식

### 2.1 Tier 1: Hough Transform (직선 차선)

**알고리즘 단계**:
1. 전처리 (Grayscale + Gaussian Blur)
2. Canny Edge Detection
3. ROI (Region of Interest) 설정
4. Hough Transform 직선 검출
5. 시각화

**장점**: 빠름, 간단함, GPU 불필요
**단점**: 곡선 처리 불가, 악천후 약함
**FPS**: 60-120

### 2.2 Tier 2: Polynomial Fitting (곡선 차선)

**알고리즘 단계**:
1. Perspective Transform (BEV 변환)
2. Histogram으로 차선 시작점 찾기
3. Sliding Window로 차선 픽셀 추출
4. Polynomial Fitting (2차 다항식)
5. Inverse Transform + 시각화

**장점**: 곡선 대응, 차선 곡률 계산 가능
**단점**: Tier 1보다 느림, 파라미터 튜닝 필요
**FPS**: 30-60

### 2.3 Tier 3: Deep Learning (LaneNet, SCNN)

**모델 비교**:

| 모델 | 구조 | FPS | 정확도 | 특징 |
|------|------|-----|--------|------|
| LaneNet | Encoder-Decoder | ~30 | 높음 | Instance Seg |
| SCNN | Slice CNN | ~15 | 매우 높음 | Spatial Info |
| Ultra-Fast | Row Anchor | ~320 | 중간 | 초고속 |
| PolyLaneNet | Polynomial | ~80 | 높음 | 곡선 적합 |

**장점**: 모든 환경 대응, 최고 정확도
**단점**: 느림, GPU 필요, 학습 데이터 필요
**FPS**: 10-30

---

## 3. 객체 탐지 및 추적

### 3.1 YOLOv8 객체 탐지

**YOLOv8 모델 크기**:

| 모델 | 파라미터 | FPS (V100) | 용도 |
|------|---------|-----------|------|
| YOLOv8n | 3.2M | ~300 | 임베디드 |
| YOLOv8s | 11.2M | ~200 | 엣지 |
| YOLOv8m | 25.9M | ~150 | 권장 |
| YOLOv8l | 43.7M | ~100 | 고성능 |
| YOLOv8x | 68.2M | ~80 | 최고정확도 |

**주요 클래스**:
- 차량 (car, truck, bus)
- 보행자 (person)
- 이륜차 (bicycle, motorcycle)
- 신호등 (traffic light)
- 표지판 (stop sign)

### 3.2 ByteTrack 객체 추적

**알고리즘**:
1. 높은 신뢰도 탐지 (conf > 0.7)
2. 낮은 신뢰도 탐지 (0.1 < conf < 0.7)
3. 1차 매칭: 높은 신뢰도 ↔ 기존 트랙
4. 2차 매칭: 낮은 신뢰도 ↔ 남은 트랙
5. 칼만 필터 예측 + 업데이트

**성능**: MOT20 기준 80.3% MOTA

### 3.3 거리 추정 (IPM)

**방법**:
1. 카메라 캘리브레이션 (체스보드)
2. Perspective Transform (IPM)
3. 픽셀 좌표 → 월드 좌표 변환

**간단한 근사**:
```python
distance = (image_height - horizon_y) / (bbox_bottom_y - horizon_y) * scale
```

---

## 4. 통합 파이프라인

### 4.1 전체 구조

```
입력 영상
    ↓
┌─────────────────────────────────────┐
│ 병렬 처리                            │
│  ├─ 차선 인식 (3-Tier)               │
│  └─ 객체 탐지 + 추적 (YOLO + ByteTrack)│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 위험도 분석                          │
│  ├─ 차선 이탈 위험도 (0~1)           │
│  ├─ 충돌 위험도 (TTC)                │
│  └─ 급정거 차량 감지                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 의사결정                             │
│  ├─ STEER_BACK (차선 복귀)          │
│  ├─ EMERGENCY_BRAKE (긴급 제동)      │
│  └─ SLOW_DOWN (감속)                │
└─────────────────────────────────────┘
    ↓
시각화 (2D/3D BEV)
```

### 4.2 위험도 분석

**TTC (Time To Collision)**:
```
TTC = distance / relative_velocity

위험도:
- TTC < 2초: HIGH (긴급 제동)
- 2초 ≤ TTC < 5초: MEDIUM (감속)
- TTC ≥ 5초: LOW (정상)
```

**차선 이탈 위험도**:
```
lateral_offset = |차량 중심 - 차선 중심|
risk = lateral_offset / lane_width

위험도:
- risk > 0.7: HIGH (차선 복귀)
- 0.4 ≤ risk ≤ 0.7: MEDIUM (경고)
- risk < 0.4: LOW (정상)
```

---

## 5. 실전 배포

### 5.1 TensorRT 최적화

**성능 향상**:
- YOLOv8m: ~150 FPS → ~300 FPS (2배)
- 메모리 사용량 50% 감소

**변환 과정**:
```bash
# 1. ONNX 변환
yolo export model=yolov8m.pt format=onnx

# 2. TensorRT 엔진 빌드
yolo export model=yolov8m.pt format=engine

# 3. 추론
model = YOLO('yolov8m.engine')
results = model(frame)
```

### 5.2 Edge 디바이스 배포

**하드웨어 선택**:

| 디바이스 | 성능 | 가격 | FPS (YOLOv8n) |
|---------|------|------|---------------|
| Raspberry Pi 4 | CPU | $50 | ~5 |
| Jetson Nano | GPU | $100 | ~20 |
| Jetson Xavier NX | GPU | $400 | ~60 |
| Jetson AGX Orin | GPU | $2000 | ~150 |

### 5.3 멀티스레딩

```python
import threading
from queue import Queue

def camera_thread(queue):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        queue.put(frame)

def inference_thread(queue, results_queue):
    model = YOLO('yolov8m.engine')
    while True:
        frame = queue.get()
        results = model(frame)
        results_queue.put(results)

# 실행
frame_queue = Queue(maxsize=10)
results_queue = Queue(maxsize=10)

t1 = threading.Thread(target=camera_thread, args=(frame_queue,))
t2 = threading.Thread(target=inference_thread, args=(frame_queue, results_queue))

t1.start()
t2.start()
```

---

## 6. 과제

### 실습 과제
1. **Tier 1-3 차선 인식 비교**: 각 방법의 FPS와 정확도 측정
2. **YOLOv8 + ByteTrack**: 실시간 객체 추적 구현
3. **거리 추정**: 카메라 캘리브레이션 및 IPM 적용
4. **통합 시스템**: 위험도 분석 및 의사결정 구현
5. **TensorRT 최적화**: 성능 비교 실험

### 프로젝트 과제
- 실제 도로 영상으로 완전한 자율주행 인식 시스템 구현
- 3가지 이상의 위험 시나리오 대응
- 성능 벤치마크 리포트 작성

---

## 참고 자료

**데이터셋**:
- TuSimple Lane Detection Dataset
- CULane Dataset
- BDD100K Dataset
- KITTI Dataset

**논문**:
- YOLOv8: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- ByteTrack: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)
- LaneNet: [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)

**프레임워크**:
- OpenCV
- PyTorch
- TensorRT
- ONNX Runtime
