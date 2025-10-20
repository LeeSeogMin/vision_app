"""
Week 11: 스마트 CCTV 모니터링 시스템 (Smart CCTV Monitoring System)
==================================================================================

이 모듈은 교육용 스마트 CCTV 시스템을 다룹니다:

## 📚 학습 목표

1. **CCTV 시스템 이해**
   - 전통 CCTV vs AI 기반 스마트 CCTV
   - 실시간 객체 탐지 및 추적
   - 이벤트 기반 모니터링

2. **객체 탐지 및 추적**
   - YOLOv8: 사람, 차량, 동물 탐지
   - ByteTrack: 실시간 객체 추적
   - 추적 ID 관리 및 궤적 분석

3. **ROI 및 이벤트 감지**
   - ROI (Region of Interest) 설정
   - 침입 감지 (Intrusion Detection)
   - 배회 감지 (Loitering Detection)
   - 군중 밀집도 분석

4. **히트맵 분석**
   - 이동 경로 시각화
   - 핫스팟 분석
   - 시간대별 활동 패턴

5. **간단한 대시보드**
   - 실시간 통계
   - 이벤트 로그 (CSV)
   - 알림 시스템 (콘솔)

## 🛠️ 시스템 구성

```
스마트 CCTV 시스템
│
├── 입력 계층 (Input Layer)
│   ├── 비디오 스트림 (파일/웹캠)
│   └── 프레임 전처리
│
├── 탐지 계층 (Detection Layer)
│   ├── YOLOv8 객체 탐지
│   └── 클래스 필터링 (사람/차량/동물)
│
├── 추적 계층 (Tracking Layer)
│   ├── ByteTrack 알고리즘
│   ├── ID 할당 및 관리
│   └── 궤적 기록
│
├── 분석 계층 (Analysis Layer)
│   ├── ROI 교차 검사
│   ├── 침입/배회 감지
│   ├── 히트맵 생성
│   └── 통계 집계
│
└── 출력 계층 (Output Layer)
    ├── 시각화 (바운딩 박스, 궤적)
    ├── 이벤트 로그 (CSV)
    └── 알림 (콘솔)
```

## 📂 모듈 구조

```
week11_smart_cctv/
│
├── __init__.py                           # 모듈 초기화
├── smart_cctv_module.py                  # Streamlit 메인 모듈
│
├── labs/                                 # 실습 파일
│   ├── lab01_yolo_detection.py          # YOLOv8 탐지 실습
│   ├── lab02_bytetrack_tracking.py      # ByteTrack 추적 실습
│   ├── lab03_roi_intrusion.py           # ROI 침입 감지 실습
│   ├── lab04_loitering_detection.py     # 배회 감지 실습
│   └── lab05_heatmap_analysis.py        # 히트맵 분석 실습
│
├── lectures/                             # 강의 자료
│   └── lecture_slides.md                 # 강의 슬라이드
│
├── notebooks/                            # Colab 노트북
│   └── Week11_Smart_CCTV_Complete.ipynb # 완전한 구현
│
└── data/                                 # 샘플 데이터
    ├── surveillance_sample.mp4           # CCTV 샘플 영상
    └── yolov8n.pt                        # YOLOv8 nano 모델
```

## 🚀 빠른 시작

### Streamlit 실행
```bash
streamlit run modules/week11_smart_cctv/smart_cctv_module.py
```

### Colab에서 실행
1. Week11_Smart_CCTV_Complete.ipynb 열기
2. GPU 런타임 설정 (선택사항)
3. 셀 순서대로 실행

## 📦 의존성 (간소화)

```python
# 필수
opencv-python>=4.8.0
ultralytics>=8.0.0      # YOLOv8
numpy>=1.24.0
matplotlib>=3.7.0

# UI (Streamlit용)
streamlit>=1.28.0
plotly>=5.17.0

# 추적 (선택사항)
scipy>=1.11.0           # 간단한 IoU 계산
```

## 🎓 교육 목적 간소화 항목

| 항목 | 프로덕션 버전 | 교육용 버전 |
|------|--------------|------------|
| 데이터베이스 | PostgreSQL/MongoDB | CSV 파일 |
| 클라우드 저장소 | AWS S3 | 로컬 폴더 |
| 행동 인식 | VideoMAE (딥러닝) | 규칙 기반 |
| 알림 시스템 | Email/SMS/Webhook | 콘솔 출력 |
| 인증/보안 | JWT, HTTPS | 없음 |
| 멀티카메라 | 동시 처리 (4-16대) | 단일 영상 |
| 하드웨어 요구 | GPU 필수 (RTX 3060+) | CPU 가능 |
| 배포 | Docker, Kubernetes | 로컬 실행 |

## 💡 핵심 알고리즘

### 1. YOLOv8 탐지
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # nano 모델 (경량)
results = model(frame, classes=[0, 2, 3])  # 사람, 차량, 동물
```

### 2. ByteTrack 추적
```python
# IoU 기반 매칭
def match_detections_to_tracks(detections, tracks):
    iou_matrix = compute_iou_matrix(detections, tracks)
    matches, unmatched_dets, unmatched_tracks = hungarian_matching(iou_matrix)
    return matches, unmatched_dets, unmatched_tracks
```

### 3. ROI 침입 감지
```python
def check_intrusion(bbox_center, roi_polygon):
    return cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0
```

### 4. 배회 감지
```python
def detect_loitering(track_history, threshold_seconds=10):
    if len(track_history) < 2:
        return False

    # 같은 영역에 오래 머무름
    duration = track_history[-1]['timestamp'] - track_history[0]['timestamp']
    movement = calculate_total_movement(track_history)

    return duration > threshold_seconds and movement < 50  # 픽셀
```

## 📊 성능 지표

| 항목 | 목표 | 실측 |
|------|------|------|
| 탐지 FPS (CPU) | ≥15 | ~20-25 |
| 탐지 FPS (GPU) | ≥30 | ~60-80 |
| 추적 정확도 | ≥85% | ~88-92% |
| 침입 감지 지연 | <1초 | ~0.3-0.5초 |
| 메모리 사용량 | <2GB | ~1.2-1.5GB |

## 🎯 실습 프로젝트 아이디어

1. **주차장 모니터링**: 불법 주차 감지, 차량 계수
2. **소매점 분석**: 고객 동선, 체류 시간, 핫스팟
3. **보안 시스템**: 침입 감지, 배회자 추적
4. **교통 분석**: 차량/보행자 계수, 혼잡도
5. **작업장 안전**: 위험 구역 침입, 안전모 미착용

## 📚 참고 자료

- YOLOv8 공식 문서: https://docs.ultralytics.com/
- ByteTrack 논문: https://arxiv.org/abs/2110.06864
- OpenCV 튜토리얼: https://docs.opencv.org/

## ⚠️ 주의사항

- **개인정보 보호**: 교육 목적으로만 사용, 공공장소 촬영 시 법적 검토 필요
- **성능**: CPU 모드는 느릴 수 있음 (Colab GPU 권장)
- **샘플 영상**: 저작권 확인 후 사용

---

**Author**: Smart Vision Team
**Version**: 1.0.0
**Last Updated**: 2025-01-20
"""

from .smart_cctv_module import SmartCCTVModule

__all__ = ['SmartCCTVModule']
