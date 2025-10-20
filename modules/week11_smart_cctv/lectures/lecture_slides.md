# Week 11: 스마트 CCTV 모니터링 시스템

## 강의 개요
- **목표**: 교육용 스마트 CCTV 시스템 이해 및 구현
- **기간**: 2주 (이론 1주 + 실습 1주)
- **난이도**: 중급

---

## Section 1: 스마트 CCTV 시스템 소개

### 1.1 전통 CCTV vs AI 기반 CCTV

| 항목 | 전통 CCTV | AI 스마트 CCTV |
|------|-----------|----------------|
| **목적** | 녹화 및 사후 확인 | 실시간 자동 분석 |
| **모니터링** | 사람 필요 (24시간) | 자동 (알림 기반) |
| **분석** | 수동 (영상 재생) | 자동 (객체 탐지/추적) |
| **반응** | 사후 조치 | 실시간 예방 |

### 1.2 시스템 구성요소

```
입력 → 탐지 → 추적 → 분석 → 출력
 ↓      ↓      ↓      ↓      ↓
영상   YOLOv8 ByteTrack  ROI  대시보드
```

### 1.3 주요 활용 분야
- 보안: 침입/배회 감지
- 교통: 차량 계수, 혼잡도
- 상업: 고객 동선, 핫스팟
- 안전: 작업장 위험 구역

---

## Section 2: YOLOv8 객체 탐지

### 2.1 YOLO 개요
- **You Only Look Once**: 한 번의 forward pass로 탐지
- **실시간**: 30-140 FPS
- **정확도**: COCO mAP 50% (YOLOv8m)

### 2.2 YOLOv8 모델 크기

| 모델 | 파라미터 | FPS (CPU) | FPS (GPU) | 용도 |
|------|---------|-----------|-----------|------|
| YOLOv8n | 3.2M | ~25 | ~140 | 실시간 |
| YOLOv8s | 11.2M | ~15 | ~90 | 균형 |
| YOLOv8m | 25.9M | ~8 | ~60 | 정확도 |

### 2.3 기본 사용법

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model(frame, conf=0.5, classes=[0, 2])  # 사람, 차량

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = int(box.cls[0])
```

---

## Section 3: ByteTrack 객체 추적

### 3.1 ByteTrack 핵심 아이디어

**2-Stage Matching**:
1. **High Score** (conf >= 0.5) → Track 매칭
2. **Low Score** (0.1 <= conf < 0.5) → 미매칭 Track 복구

### 3.2 IoU 기반 매칭

```python
def calculate_iou(box1, box2):
    inter_area = intersection(box1, box2)
    union_area = area(box1) + area(box2) - inter_area
    return inter_area / union_area

# IoU >= 0.3이면 같은 객체로 판단
```

### 3.3 Kalman Filter 예측

```python
# 등속 모델 (교육용 간소화)
velocity_x = history[-1].x - history[-2].x
velocity_y = history[-1].y - history[-2].y

predicted_x = history[-1].x + velocity_x
predicted_y = history[-1].y + velocity_y
```

---

## Section 4: ROI 및 이벤트 감지

### 4.1 ROI (Region of Interest)

```python
roi_polygon = np.array([
    [100, 200],  # 좌상
    [400, 200],  # 우상
    [450, 400],  # 우하
    [50, 400]    # 좌하
])

# 점이 ROI 내부인지 확인
is_inside = cv2.pointPolygonTest(roi_polygon, center, False) >= 0
```

### 4.2 침입 감지 (Intrusion)

**조건**: ROI 내부 + 체류 시간 > 3초

```python
if is_in_roi and duration > threshold:
    alert('INTRUSION')
```

### 4.3 배회 감지 (Loitering)

**조건**: 체류 시간 길고 + 이동 거리 짧음

```python
if duration > 10sec and movement < 100px:
    alert('LOITERING')
```

---

## Section 5: 히트맵 분석

### 5.1 히트맵 생성

```python
# 1. 시간 감쇠
heatmap *= 0.995

# 2. 현재 Track 위치 누적
for track in tracks:
    center = track.get_center()
    heatmap[center_y, center_x] += 1.0

# 3. Gaussian 블러
cv2.GaussianBlur(heatmap, (21, 21), 0)

# 4. 컬러맵 적용
heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
```

### 5.2 핫스팟 추출

```python
# 상위 10% 영역
threshold = np.percentile(heatmap, 90)
hotspot_mask = (heatmap > threshold)

# 연결 컴포넌트 분석
contours, _ = cv2.findContours(hotspot_mask, ...)
```

---

## Section 6: 대시보드 및 로깅

### 6.1 실시간 메트릭

- 현재 추적 중인 객체 수
- 총 탐지 횟수
- 이벤트 발생 횟수 (침입/배회)
- 평균 FPS

### 6.2 CSV 로깅

```python
# events_20250120.csv
Timestamp, Event Type, Track ID, Position, Duration, Message
2025-01-20 10:30:15, INTRUSION, 42, (320, 240), 5.2, Track 42 in ROI
```

### 6.3 알림 시스템 (교육용)

```python
# 콘솔 출력
print(f"[ALERT] {event['message']}")

# Streamlit 토스트
st.toast(event['message'], icon="🚨")
```

---

## Section 7: 실습 프로젝트

### 프로젝트 1: 주차장 모니터링
- 불법 주차 감지
- 차량 계수
- 주차 공간 점유율

### 프로젝트 2: 소매점 분석
- 고객 동선 분석
- 체류 시간 측정
- 인기 구역 파악

### 프로젝트 3: 보안 시스템
- 침입 자동 감지
- 배회자 추적
- 실시간 알림

---

## Section 8: 성능 최적화

### 8.1 속도 향상

```python
# 1. 경량 모델
model = YOLO('yolov8n.pt')

# 2. 이미지 크기 축소
results = model(frame, imgsz=640)

# 3. 프레임 스킵
if frame_idx % 2 == 0:
    results = model(frame)
```

### 8.2 정확도 향상

```python
# 1. 신뢰도 임계값
results = model(frame, conf=0.5)

# 2. NMS 임계값
results = model(frame, iou=0.5)

# 3. 클래스 필터링
results = model(frame, classes=[0, 2])
```

---

## Section 9: 프로덕션 배포 (참고)

### 9.1 교육용 vs 프로덕션

| 항목 | 교육용 | 프로덕션 |
|------|--------|----------|
| DB | CSV | PostgreSQL/MongoDB |
| 저장소 | 로컬 | AWS S3 |
| 알림 | 콘솔 | Email/SMS/Webhook |
| 카메라 | 1대 | 4-16대 동시 |
| 인증 | 없음 | JWT, HTTPS |

### 9.2 클라우드 배포 (AWS)

- EC2: 추론 서버 (g4dn.xlarge)
- S3: 영상 저장
- RDS: 메타데이터
- Lambda: 이벤트 처리
- CloudWatch: 모니터링

**월간 비용**: ~$400

### 9.3 Edge AI 배포

- NVIDIA Jetson Orin Nano ($500)
- 로컬 추론 (30 FPS)
- S3 영상 업로드 (선택적)

**월간 비용**: ~$50

---

## 참고 자료

### 공식 문서
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [OpenCV Tutorials](https://docs.opencv.org/)

### 데이터셋
- COCO: http://cocodataset.org/
- MOT Challenge: https://motchallenge.net/

### 도구
- Roboflow: 데이터셋 관리
- WandB: 실험 추적
- TensorBoard: 시각화

---

## 과제

### 과제 1: 기본 구현 (60점)
- YOLOv8 + ByteTrack 통합
- ROI 침입 감지
- 히트맵 생성
- CSV 로깅

### 과제 2: 고급 기능 (30점)
- 배회 감지 구현
- 시간대별 통계
- 핫스팟 분석

### 과제 3: 프로젝트 (10점)
- 실제 활용 사례 구현
- 발표 자료 (5분)

---

## Q&A

**Q1: GPU 없이도 실행 가능한가요?**
A: 네, YOLOv8n은 CPU에서도 15-25 FPS 가능합니다.

**Q2: 실제 CCTV 카메라 연결 방법은?**
A: RTSP URL 사용: `cv2.VideoCapture('rtsp://username:password@ip:port/stream')`

**Q3: 여러 대 카메라 동시 처리는?**
A: 멀티스레딩 또는 멀티프로세싱 사용. 프로덕션 예제는 Colab 참고.

**Q4: 클라우드 비용 절감 방법은?**
A: Edge AI (Jetson) 사용으로 80% 이상 절감 가능.

---

**강의 종료**

교육 목적으로 구현한 스마트 CCTV 시스템을 실제 프로젝트에 적용할 때는:
- 개인정보 보호법 확인
- 보안 강화 (인증, 암호화)
- 확장성 고려 (멀티카메라, 클라우드)
- 모니터링 시스템 구축

**Good Luck! 🚀**
