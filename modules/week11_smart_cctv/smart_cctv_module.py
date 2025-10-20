"""
Week 11: Smart CCTV Monitoring System
교육용 스마트 CCTV 모니터링 시스템

Author: Smart Vision Team
Date: 2025-01-20
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
import time
from collections import deque, defaultdict
import csv
from datetime import datetime

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.base_image_processor import BaseImageProcessor


class SmartCCTVModule(BaseImageProcessor):
    """
    스마트 CCTV 모니터링 시스템 메인 모듈

    교육용으로 간소화된 CCTV 시스템:
    - YOLOv8 객체 탐지
    - ByteTrack 추적
    - ROI 기반 이벤트 감지
    - 히트맵 분석
    - 간단한 대시보드
    """

    def __init__(self):
        super().__init__()
        self.name = "Week 11: Smart CCTV System"

        # 추적 관련 상태
        if 'tracks' not in st.session_state:
            st.session_state.tracks = {}  # {track_id: track_info}
        if 'next_track_id' not in st.session_state:
            st.session_state.next_track_id = 1
        if 'heatmap' not in st.session_state:
            st.session_state.heatmap = None
        if 'event_log' not in st.session_state:
            st.session_state.event_log = []
        if 'roi_points' not in st.session_state:
            st.session_state.roi_points = []

    def render(self):
        """메인 렌더링 함수 - 5개 탭"""
        st.title("🎥 Week 11: Smart CCTV Monitoring System")

        st.markdown("""
        ### 교육용 스마트 CCTV 시스템

        **핵심 기능**:
        - 🎯 YOLOv8 실시간 탐지 (사람/차량/동물)
        - 🔍 ByteTrack 객체 추적
        - 🚨 ROI 기반 이벤트 감지 (침입/배회)
        - 🔥 히트맵 분석 (경로 시각화)
        - 📊 간단한 대시보드 (통계/로그)
        """)

        # 5개 탭 생성
        tabs = st.tabs([
            "📚 1. CCTV 시스템 이론",
            "🎯 2. 탐지 및 추적",
            "🚨 3. ROI 및 이벤트",
            "🔥 4. 히트맵 분석",
            "📊 5. 대시보드"
        ])

        with tabs[0]:
            self.render_theory()

        with tabs[1]:
            self.render_detection_tracking()

        with tabs[2]:
            self.render_roi_events()

        with tabs[3]:
            self.render_heatmap()

        with tabs[4]:
            self.render_dashboard()

    def render_theory(self):
        """Tab 1: CCTV 시스템 이론"""
        st.header("📚 스마트 CCTV 시스템 이론")

        st.markdown("---")

        # 1. 전통 CCTV vs AI 기반 스마트 CCTV
        st.subheader("1️⃣ 전통 CCTV vs AI 기반 스마트 CCTV")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🎬 전통 CCTV

            **특징**:
            - 영상 녹화 및 저장
            - 사람이 직접 모니터링
            - 사후 확인 중심
            - 단순 움직임 감지

            **한계**:
            - 24시간 모니터링 인력 필요
            - 사건 발생 후 확인
            - 정확한 분석 어려움
            - 대량 영상 검색 시간 소요

            **활용**:
            - 기본 보안
            - 증거 자료
            - 사후 조사
            """)

        with col2:
            st.markdown("""
            #### 🤖 AI 기반 스마트 CCTV

            **특징**:
            - 실시간 객체 탐지 및 추적
            - 자동 이벤트 감지
            - 사전 예방 중심
            - 지능형 알림 시스템

            **장점**:
            - 실시간 자동 모니터링
            - 즉시 이벤트 알림
            - 정확한 객체 인식
            - 빠른 검색 및 분석

            **활용**:
            - 침입 감지
            - 교통 분석
            - 군중 관리
            - 작업장 안전
            """)

        st.info("💡 **교육 포인트**: AI 기반 CCTV는 '사후 확인'에서 '사전 예방'으로 패러다임 전환")

        # 2. 시스템 구성요소
        st.markdown("---")
        st.subheader("2️⃣ 스마트 CCTV 시스템 구성")

        st.code("""
스마트 CCTV 시스템 아키텍처
═══════════════════════════

┌─────────────────────────────────────────────────┐
│          입력 계층 (Input Layer)                 │
│  • 비디오 스트림 (파일/웹캠/IP카메라)             │
│  • 프레임 전처리 (리사이즈, 정규화)               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        탐지 계층 (Detection Layer)               │
│  • YOLOv8: 사람, 차량, 동물 탐지                 │
│  • 바운딩 박스 + 신뢰도 점수                     │
│  • 클래스 필터링                                  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        추적 계층 (Tracking Layer)                │
│  • ByteTrack: ID 할당 및 추적                    │
│  • Kalman Filter: 위치 예측                      │
│  • 궤적 기록 (Trajectory)                        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        분석 계층 (Analysis Layer)                │
│  • ROI 교차 검사 (침입 감지)                     │
│  • 배회 감지 (체류 시간)                         │
│  • 히트맵 생성 (이동 경로)                       │
│  • 통계 집계 (카운팅)                            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│        출력 계층 (Output Layer)                  │
│  • 시각화 (바운딩 박스, 궤적, ROI)               │
│  • 이벤트 로그 (CSV)                             │
│  • 알림 (콘솔/이메일/SMS)                        │
└─────────────────────────────────────────────────┘
        """, language="text")

        # 3. 핵심 알고리즘 비교
        st.markdown("---")
        st.subheader("3️⃣ 핵심 알고리즘")

        st.markdown("#### 📊 객체 탐지 모델 비교")

        comparison_data = {
            "모델": ["YOLOv8n", "YOLOv8s", "YOLOv8m", "Faster R-CNN", "SSD"],
            "파라미터": ["3.2M", "11.2M", "25.9M", "41.8M", "23.5M"],
            "정확도 (mAP)": ["37.3%", "44.9%", "50.2%", "42.0%", "25.1%"],
            "FPS (CPU)": ["~25", "~15", "~8", "~5", "~10"],
            "FPS (GPU)": ["~140", "~90", "~60", "~20", "~45"],
            "용도": ["실시간", "균형", "정확도", "정밀", "경량"]
        }

        st.table(comparison_data)

        st.info("💡 **교육 선택**: YOLOv8n (nano) - 빠른 속도 + 적절한 정확도")

        st.markdown("#### 🔍 객체 추적 알고리즘 비교")

        tracking_data = {
            "알고리즘": ["ByteTrack", "DeepSORT", "SORT", "CenterTrack"],
            "추적 방식": ["Detection-based", "Detection + ReID", "Detection-based", "Detection-based"],
            "정확도": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐"],
            "속도": ["⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐"],
            "복잡도": ["중간", "높음", "낮음", "중간"],
            "특징": ["고속 + 고정확도", "외형 특징 사용", "매우 간단", "추적 + 탐지 동시"]
        }

        st.table(tracking_data)

        st.info("💡 **교육 선택**: ByteTrack (간소화) - 최신 알고리즘, 높은 성능")

        # 4. 주요 활용 분야
        st.markdown("---")
        st.subheader("4️⃣ 스마트 CCTV 활용 분야")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            #### 🏢 보안 및 안전

            - **침입 감지**: 금지 구역 침입
            - **배회 감지**: 의심스러운 행동
            - **폭력 감지**: 싸움, 쓰러짐
            - **화재/연기 감지**: 재난 대응

            **적용 사례**:
            - 은행, 관공서
            - 주택, 아파트
            - 공장, 창고
            """)

        with col2:
            st.markdown("""
            #### 🚗 교통 및 도시

            - **차량 계수**: 교통량 분석
            - **불법 주정차**: 자동 단속
            - **사고 감지**: 즉시 대응
            - **혼잡도 분석**: 교통 최적화

            **적용 사례**:
            - 도로, 교차로
            - 주차장
            - 톨게이트
            """)

        with col3:
            st.markdown("""
            #### 🛍️ 상업 및 분석

            - **고객 동선**: 매장 배치 최적화
            - **대기 시간**: 계산대 인력 배치
            - **핫스팟 분석**: 인기 구역 파악
            - **재고 관리**: 진열대 모니터링

            **적용 사례**:
            - 소매점, 마트
            - 쇼핑몰
            - 레스토랑
            """)

        # 5. 시스템 요구사항 (간소화)
        st.markdown("---")
        st.subheader("5️⃣ 교육용 시스템 요구사항")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 💻 하드웨어

            **최소 사양** (CPU 모드):
            - CPU: Intel i5 이상
            - RAM: 8GB
            - 저장공간: 5GB
            - 웹캠 또는 샘플 영상

            **권장 사양** (GPU 모드):
            - CPU: Intel i7 이상
            - RAM: 16GB
            - GPU: NVIDIA GTX 1660 (6GB VRAM)
            - 저장공간: 10GB
            """)

        with col2:
            st.markdown("""
            #### 📦 소프트웨어

            **필수 라이브러리**:
            ```bash
            pip install opencv-python
            pip install ultralytics
            pip install numpy matplotlib
            pip install streamlit plotly
            ```

            **선택 사항**:
            - Google Colab (무료 GPU)
            - CUDA Toolkit (로컬 GPU)
            """)

        st.success("""
        ✅ **교육 목적 간소화**:
        - AWS/클라우드 불필요 → 로컬 실행
        - PostgreSQL 불필요 → CSV 로그
        - 복잡한 인증 불필요 → 단순 실행
        - 고가 장비 불필요 → 일반 노트북
        """)

    def render_detection_tracking(self):
        """Tab 2: 탐지 및 추적"""
        st.header("🎯 탐지 및 추적 (Detection & Tracking)")

        st.markdown("---")

        # YOLOv8 탐지
        st.subheader("1️⃣ YOLOv8 객체 탐지")

        st.markdown("""
        #### 🎯 YOLOv8 개요

        **YOLO (You Only Look Once)**는 실시간 객체 탐지를 위한 최신 알고리즘입니다.

        **YOLOv8 특징**:
        - **속도**: 실시간 처리 (30-140 FPS)
        - **정확도**: COCO 데이터셋 50% mAP
        - **경량화**: nano 모델 3.2M 파라미터
        - **다양성**: n/s/m/l/x 5가지 크기
        """)

        st.code("""
# YOLOv8 기본 사용법
from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('yolov8n.pt')  # nano 모델 (가장 빠름)

# 2. 추론
results = model(frame, conf=0.5)  # 신뢰도 50% 이상

# 3. 결과 추출
for result in results:
    boxes = result.boxes  # 바운딩 박스
    for box in boxes:
        # 좌표
        x1, y1, x2, y2 = box.xyxy[0]

        # 신뢰도
        confidence = box.conf[0]

        # 클래스 (0: 사람, 2: 차량, ...)
        class_id = int(box.cls[0])

        print(f"Class: {class_id}, Conf: {confidence:.2f}")
        """, language="python")

        st.info("💡 **교육 포인트**: YOLOv8은 한 번의 forward pass로 모든 객체 탐지 (빠름)")

        # COCO 클래스
        with st.expander("📋 COCO 데이터셋 주요 클래스 (80개)"):
            st.code("""
COCO Classes (일부):
-------------------
0: person (사람)
1: bicycle (자전거)
2: car (차량)
3: motorcycle (오토바이)
5: bus (버스)
7: truck (트럭)
14: bird (새)
15: cat (고양이)
16: dog (개)

# CCTV 용도로 필터링
target_classes = [0, 2, 5, 7]  # 사람, 차량, 버스, 트럭
results = model(frame, classes=target_classes)
            """, language="python")

        # ByteTrack 추적
        st.markdown("---")
        st.subheader("2️⃣ ByteTrack 객체 추적")

        st.markdown("""
        #### 🔍 ByteTrack 개요

        **ByteTrack**는 2021년 제안된 최신 다중 객체 추적 알고리즘입니다.

        **핵심 아이디어**:
        1. **High Score Detection**: 신뢰도 높은 탐지 → Track 매칭
        2. **Low Score Detection**: 신뢰도 낮은 탐지 → 기존 Track 복구
        3. **Kalman Filter**: 위치 예측으로 가려짐(occlusion) 처리

        **장점**:
        - 가려진 객체도 추적 유지
        - 높은 정확도 (MOT17: 80.3% MOTA)
        - 실시간 처리 가능 (30 FPS)
        """)

        st.code("""
# ByteTrack 간소화 구현 (교육용)
class SimpleByteTrack:
    def __init__(self):
        self.tracks = {}  # {track_id: track_info}
        self.next_id = 1
        self.max_age = 30  # 30프레임 동안 미탐지 시 삭제

    def update(self, detections):
        \"\"\"
        detections: List[Dict]
            [{'bbox': [x1,y1,x2,y2], 'conf': 0.9, 'class': 0}, ...]
        \"\"\"

        # 1. 고신뢰도 탐지 (conf >= 0.5)
        high_dets = [d for d in detections if d['conf'] >= 0.5]

        # 2. 저신뢰도 탐지 (0.1 <= conf < 0.5)
        low_dets = [d for d in detections if 0.1 <= d['conf'] < 0.5]

        # 3. 고신뢰도 탐지와 기존 Track 매칭
        matches, unmatched_dets, unmatched_tracks = self.match(high_dets, self.tracks)

        # 4. 매칭된 Track 업데이트
        for det_idx, track_id in matches:
            self.tracks[track_id].update(high_dets[det_idx])

        # 5. 미매칭 Track과 저신뢰도 탐지 재매칭
        matches2, unmatched_low, still_unmatched = self.match(low_dets, unmatched_tracks)

        for det_idx, track_id in matches2:
            self.tracks[track_id].update(low_dets[det_idx])

        # 6. 새로운 Track 생성
        for det_idx in unmatched_dets:
            self.tracks[self.next_id] = Track(self.next_id, high_dets[det_idx])
            self.next_id += 1

        # 7. 오래된 Track 삭제
        self.remove_old_tracks()

        return self.tracks

    def match(self, detections, tracks):
        \"\"\"IoU 기반 헝가리안 매칭\"\"\"
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(detections))), list(tracks.keys())

        # IoU 행렬 계산
        iou_matrix = np.zeros((len(detections), len(tracks)))
        track_ids = list(tracks.keys())

        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                iou = self.calculate_iou(det['bbox'], tracks[track_id].bbox)
                iou_matrix[i, j] = iou

        # 헝가리안 알고리즘 (간소화: greedy matching)
        matches = []
        matched_dets = set()
        matched_tracks = set()

        # IoU 0.5 이상만 매칭
        threshold = 0.5
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < threshold:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matches.append((i, track_ids[j]))
            matched_dets.add(i)
            matched_tracks.add(track_ids[j])

            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]

        return matches, unmatched_dets, unmatched_tracks

    def calculate_iou(self, box1, box2):
        \"\"\"IoU (Intersection over Union) 계산\"\"\"
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 교집합
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # 합집합
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

class Track:
    \"\"\"개별 Track 클래스\"\"\"
    def __init__(self, track_id, detection):
        self.id = track_id
        self.bbox = detection['bbox']
        self.class_id = detection['class']
        self.confidence = detection['conf']

        # 궤적
        self.history = deque(maxlen=30)  # 최근 30프레임
        center = self.get_center(self.bbox)
        self.history.append(center)

        # 시간
        self.age = 0  # Track 생성 후 프레임 수
        self.time_since_update = 0  # 마지막 업데이트 후 프레임 수

    def update(self, detection):
        \"\"\"탐지 결과로 Track 업데이트\"\"\"
        self.bbox = detection['bbox']
        self.confidence = detection['conf']

        center = self.get_center(self.bbox)
        self.history.append(center)

        self.age += 1
        self.time_since_update = 0

    def predict(self):
        \"\"\"Kalman Filter 예측 (간소화: 등속 모델)\"\"\"
        if len(self.history) >= 2:
            # 속도 = 마지막 2프레임 변위
            velocity = (
                self.history[-1][0] - self.history[-2][0],
                self.history[-1][1] - self.history[-2][1]
            )

            # 예측 위치 = 현재 위치 + 속도
            pred_center = (
                self.history[-1][0] + velocity[0],
                self.history[-1][1] + velocity[1]
            )

            # bbox 업데이트
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            self.bbox = [
                pred_center[0] - w/2,
                pred_center[1] - h/2,
                pred_center[0] + w/2,
                pred_center[1] + h/2
            ]

        self.time_since_update += 1

    @staticmethod
    def get_center(bbox):
        \"\"\"bbox 중심점\"\"\"
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        """, language="python")

        st.info("💡 **교육 포인트**: ByteTrack은 저신뢰도 탐지도 활용해 가려진 객체 추적")

        # 통합 예제
        st.markdown("---")
        st.subheader("3️⃣ YOLOv8 + ByteTrack 통합")

        st.code("""
# 통합 예제 (프레임별 처리)
import cv2
from ultralytics import YOLO

# 초기화
model = YOLO('yolov8n.pt')
tracker = SimpleByteTrack()

# 비디오 처리
cap = cv2.VideoCapture('surveillance.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. YOLOv8 탐지
    results = model(frame, conf=0.3, classes=[0, 2])  # 사람, 차량

    # 2. 탐지 결과 변환
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'class': class_id
            })

    # 3. ByteTrack 추적
    tracks = tracker.update(detections)

    # 4. 시각화
    for track_id, track in tracks.items():
        x1, y1, x2, y2 = map(int, track.bbox)

        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Track ID
        label = f"ID:{track_id} {track.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 궤적
        if len(track.history) > 1:
            points = np.array(track.history, dtype=np.int32)
            cv2.polylines(frame, [points], False, (0, 0, 255), 2)

    cv2.imshow('Smart CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        """, language="python")

        st.success("""
        ✅ **통합 처리 흐름**:
        1. YOLOv8으로 프레임마다 객체 탐지
        2. ByteTrack으로 탐지 결과를 Track에 매칭
        3. 각 Track에 고유 ID 부여
        4. 궤적 시각화로 이동 경로 표시
        """)

        # 성능 최적화 팁
        st.markdown("---")
        st.subheader("4️⃣ 성능 최적화 팁")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### ⚡ 속도 향상

            ```python
            # 1. 경량 모델 사용
            model = YOLO('yolov8n.pt')  # nano

            # 2. 이미지 크기 축소
            results = model(frame, imgsz=640)  # 기본 640

            # 3. 배치 처리 (다중 카메라)
            results = model([frame1, frame2], batch=True)

            # 4. GPU 사용
            model = YOLO('yolov8n.pt').to('cuda')

            # 5. 프레임 스킵
            if frame_count % 2 == 0:  # 2프레임마다
                results = model(frame)
            ```
            """)

        with col2:
            st.markdown("""
            #### 🎯 정확도 향상

            ```python
            # 1. 신뢰도 임계값 조정
            results = model(frame, conf=0.5)  # 기본 0.25

            # 2. NMS 임계값 조정
            results = model(frame, iou=0.5)  # 기본 0.7

            # 3. 클래스 필터링
            results = model(frame, classes=[0])  # 사람만

            # 4. 이미지 전처리
            frame = cv2.GaussianBlur(frame, (5,5), 0)

            # 5. TTA (Test Time Augmentation)
            results = model(frame, augment=True)
            ```
            """)

    def render_roi_events(self):
        """Tab 3: ROI 및 이벤트 감지"""
        st.header("🚨 ROI 및 이벤트 감지")

        st.markdown("---")

        # ROI 개념
        st.subheader("1️⃣ ROI (Region of Interest) 개념")

        st.markdown("""
        #### 📍 ROI란?

        **ROI (Region of Interest)**는 관심 영역을 의미합니다.

        **CCTV에서의 활용**:
        - 침입 금지 구역 설정
        - 계수 라인 (counting line)
        - 주차 구역
        - 위험 구역

        **표현 방식**:
        - **사각형**: `[x1, y1, x2, y2]`
        - **폴리곤**: `[(x1,y1), (x2,y2), (x3,y3), ...]`
        """)

        st.code("""
# ROI 설정 예제
import cv2
import numpy as np

# 1. 사각형 ROI
roi_rect = [100, 100, 400, 300]  # x1, y1, x2, y2

def is_in_rect_roi(point, roi):
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x <= x2 and y1 <= y <= y2

# 2. 폴리곤 ROI
roi_polygon = np.array([
    [100, 200],  # 좌상
    [400, 200],  # 우상
    [450, 400],  # 우하
    [50, 400]    # 좌하
], dtype=np.int32)

def is_in_polygon_roi(point, roi):
    \"\"\"OpenCV pointPolygonTest 사용\"\"\"
    result = cv2.pointPolygonTest(roi, point, False)
    return result >= 0  # 0: 경계, 1: 내부, -1: 외부

# 3. ROI 시각화
def draw_roi(frame, roi_polygon, color=(0, 255, 0)):
    # 반투명 오버레이
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_polygon], color)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # 경계선
    cv2.polylines(frame, [roi_polygon], True, color, 2)

    return frame
        """, language="python")

        st.info("💡 **교육 포인트**: ROI는 전체 프레임이 아닌 특정 영역만 모니터링해 효율 향상")

        # 침입 감지
        st.markdown("---")
        st.subheader("2️⃣ 침입 감지 (Intrusion Detection)")

        st.markdown("""
        #### 🚨 침입 감지 알고리즘

        **정의**: 금지 구역에 객체가 진입했는지 확인

        **처리 흐름**:
        1. ROI 폴리곤 정의
        2. 객체 중심점 계산
        3. 중심점이 ROI 내부인지 검사
        4. 일정 시간 체류 시 알림
        """)

        st.code("""
# 침입 감지 구현
class IntrusionDetector:
    def __init__(self, roi_polygon, alert_threshold_seconds=3):
        self.roi = roi_polygon
        self.threshold = alert_threshold_seconds
        self.intrusion_tracks = {}  # {track_id: first_intrusion_time}

    def check_intrusion(self, tracks, current_time):
        \"\"\"
        tracks: Dict[int, Track] - 현재 프레임의 모든 Track
        current_time: float - 현재 시간 (초)
        \"\"\"
        alerts = []

        for track_id, track in tracks.items():
            # 중심점 계산
            center = track.get_center(track.bbox)

            # ROI 내부 검사
            is_inside = cv2.pointPolygonTest(self.roi, center, False) >= 0

            if is_inside:
                # 처음 침입한 경우
                if track_id not in self.intrusion_tracks:
                    self.intrusion_tracks[track_id] = current_time

                # 체류 시간 계산
                duration = current_time - self.intrusion_tracks[track_id]

                # 임계값 초과 시 알림
                if duration >= self.threshold:
                    alerts.append({
                        'type': 'INTRUSION',
                        'track_id': track_id,
                        'duration': duration,
                        'position': center,
                        'message': f'Track {track_id} in ROI for {duration:.1f}s'
                    })
            else:
                # ROI 외부로 나간 경우 기록 삭제
                if track_id in self.intrusion_tracks:
                    del self.intrusion_tracks[track_id]

        return alerts

    def draw_intrusion_overlay(self, frame, tracks):
        \"\"\"침입 시각화\"\"\"
        # ROI 그리기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.roi], (0, 0, 255))  # 빨간색
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.polylines(frame, [self.roi], True, (0, 0, 255), 3)

        # 침입 중인 Track 강조
        for track_id in self.intrusion_tracks:
            if track_id in tracks:
                track = tracks[track_id]
                x1, y1, x2, y2 = map(int, track.bbox)

                # 빨간색 바운딩 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                # 경고 텍스트
                cv2.putText(frame, "INTRUSION!", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

# 사용 예제
roi = np.array([[100, 200], [400, 200], [450, 400], [50, 400]])
detector = IntrusionDetector(roi, alert_threshold_seconds=3)

# 프레임별 처리
for frame_idx, frame in enumerate(video_frames):
    # YOLOv8 + ByteTrack
    detections = yolo_model(frame)
    tracks = tracker.update(detections)

    # 침입 감지
    current_time = frame_idx / fps  # 초 단위
    alerts = detector.check_intrusion(tracks, current_time)

    # 알림 처리
    for alert in alerts:
        print(f"[ALERT] {alert['message']}")
        log_to_csv(alert)
        # send_notification(alert)  # 이메일/SMS

    # 시각화
    frame = detector.draw_intrusion_overlay(frame, tracks)
    cv2.imshow('Intrusion Detection', frame)
        """, language="python")

        st.success("✅ **침입 감지 핵심**: ROI 내부 체류 시간으로 오탐 (false positive) 최소화")

        # 배회 감지
        st.markdown("---")
        st.subheader("3️⃣ 배회 감지 (Loitering Detection)")

        st.markdown("""
        #### 👀 배회 감지 알고리즘

        **정의**: 특정 영역에서 오랜 시간 머물거나 반복적으로 왔다갔다 하는 행동 감지

        **처리 흐름**:
        1. Track 궤적 기록 (최근 N프레임)
        2. 이동 거리 계산
        3. 체류 시간 계산
        4. 이동 거리 작고 + 체류 시간 긴 경우 배회로 판단
        """)

        st.code("""
# 배회 감지 구현
class LoiteringDetector:
    def __init__(self, min_duration_seconds=10, max_movement_pixels=100):
        self.min_duration = min_duration_seconds
        self.max_movement = max_movement_pixels
        self.loitering_tracks = {}  # {track_id: start_time}

    def check_loitering(self, tracks, current_time, fps=30):
        \"\"\"
        tracks: Dict[int, Track]
        current_time: float - 초 단위
        fps: int - 프레임 레이트
        \"\"\"
        alerts = []

        for track_id, track in tracks.items():
            # 궤적이 충분히 쌓인 경우만 판단
            min_frames = int(self.min_duration * fps)
            if len(track.history) < min_frames:
                continue

            # 최근 N초 동안의 이동 거리 계산
            recent_history = list(track.history)[-min_frames:]
            total_movement = self.calculate_total_movement(recent_history)

            # 배회 조건: 이동 거리 작음 + 체류 시간 김
            if total_movement < self.max_movement:
                # 처음 배회로 판단된 경우
                if track_id not in self.loitering_tracks:
                    self.loitering_tracks[track_id] = current_time

                # 배회 시간 계산
                duration = current_time - self.loitering_tracks[track_id]

                alerts.append({
                    'type': 'LOITERING',
                    'track_id': track_id,
                    'duration': duration,
                    'movement': total_movement,
                    'position': track.history[-1],
                    'message': f'Track {track_id} loitering for {duration:.1f}s (moved {total_movement:.0f}px)'
                })
            else:
                # 이동 거리가 크면 배회 아님
                if track_id in self.loitering_tracks:
                    del self.loitering_tracks[track_id]

        return alerts

    def calculate_total_movement(self, history):
        \"\"\"궤적의 총 이동 거리 (픽셀)\"\"\"
        if len(history) < 2:
            return 0

        total = 0
        for i in range(1, len(history)):
            p1 = history[i-1]
            p2 = history[i]
            distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total += distance

        return total

    def draw_loitering_overlay(self, frame, tracks):
        \"\"\"배회 시각화\"\"\"
        for track_id in self.loitering_tracks:
            if track_id in tracks:
                track = tracks[track_id]
                x1, y1, x2, y2 = map(int, track.bbox)

                # 주황색 바운딩 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)

                # 경고 텍스트
                cv2.putText(frame, "LOITERING!", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # 궤적 그리기
                if len(track.history) > 1:
                    points = np.array(track.history, dtype=np.int32)
                    cv2.polylines(frame, [points], False, (0, 165, 255), 2)

        return frame

# 사용 예제
detector = LoiteringDetector(min_duration_seconds=10, max_movement_pixels=100)

for frame_idx, frame in enumerate(video_frames):
    # YOLOv8 + ByteTrack
    detections = yolo_model(frame)
    tracks = tracker.update(detections)

    # 배회 감지
    current_time = frame_idx / fps
    alerts = detector.check_loitering(tracks, current_time, fps)

    # 알림 처리
    for alert in alerts:
        print(f"[ALERT] {alert['message']}")

    # 시각화
    frame = detector.draw_loitering_overlay(frame, tracks)
    cv2.imshow('Loitering Detection', frame)
        """, language="python")

        st.info("💡 **교육 포인트**: 배회 = 체류 시간 길고 + 이동 거리 짧음")

        # 추가 이벤트
        st.markdown("---")
        st.subheader("4️⃣ 기타 이벤트 감지")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📊 객체 계수 (Counting)

            ```python
            class LineCrossingCounter:
                def __init__(self, line_start, line_end):
                    self.line = (line_start, line_end)
                    self.crossed_tracks = set()
                    self.count = 0

                def check_crossing(self, tracks):
                    for track_id, track in tracks.items():
                        if track_id in self.crossed_tracks:
                            continue

                        if len(track.history) >= 2:
                            p1 = track.history[-2]
                            p2 = track.history[-1]

                            # 선분 교차 검사
                            if self.line_intersect(p1, p2):
                                self.count += 1
                                self.crossed_tracks.add(track_id)

                    return self.count

                def line_intersect(self, p1, p2):
                    # CCW 알고리즘
                    # ...
                    pass
            ```
            """)

        with col2:
            st.markdown("""
            #### 🚗 속도 측정

            ```python
            def estimate_speed(track, fps, pixels_per_meter):
                \"\"\"
                track: Track 객체
                fps: 프레임 레이트
                pixels_per_meter: 픽셀-미터 변환
                \"\"\"
                if len(track.history) < 2:
                    return 0

                # 최근 2프레임 변위 (픽셀)
                p1 = track.history[-2]
                p2 = track.history[-1]
                displacement_px = np.sqrt(
                    (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
                )

                # 거리 (미터)
                distance_m = displacement_px / pixels_per_meter

                # 시간 (초)
                time_s = 1 / fps

                # 속도 (m/s)
                speed_ms = distance_m / time_s

                # km/h 변환
                speed_kmh = speed_ms * 3.6

                return speed_kmh
            ```
            """)

    def render_heatmap(self):
        """Tab 4: 히트맵 분석"""
        st.header("🔥 히트맵 분석 (Heatmap Analysis)")

        st.markdown("---")

        # 히트맵 개념
        st.subheader("1️⃣ 히트맵이란?")

        st.markdown("""
        #### 🌡️ 히트맵 (Heatmap) 개요

        **정의**: 객체의 이동 경로 및 체류 시간을 색상으로 시각화

        **활용**:
        - **소매점**: 고객 동선, 인기 구역 파악
        - **교통**: 혼잡 구역 분석
        - **보안**: 활동 빈도 높은 영역
        - **작업장**: 위험 구역 출입 빈도

        **색상 매핑**:
        - 🔵 파란색: 낮은 활동
        - 🟢 초록색: 중간 활동
        - 🟡 노란색: 높은 활동
        - 🔴 빨간색: 매우 높은 활동
        """)

        st.code("""
# 히트맵 생성 구현
class HeatmapGenerator:
    def __init__(self, frame_shape, decay_factor=0.99):
        \"\"\"
        frame_shape: (height, width) - 프레임 크기
        decay_factor: float - 시간에 따른 감쇠 (0.99 = 천천히 사라짐)
        \"\"\"
        self.height, self.width = frame_shape

        # 히트맵 누적 배열 (float32)
        self.heatmap = np.zeros((self.height, self.width), dtype=np.float32)

        self.decay_factor = decay_factor

    def update(self, tracks):
        \"\"\"Track 위치를 히트맵에 누적\"\"\"

        # 1. 시간 감쇠 (오래된 정보 점차 사라짐)
        self.heatmap *= self.decay_factor

        # 2. 현재 Track 위치 누적
        for track_id, track in tracks.items():
            center = track.get_center(track.bbox)
            x, y = map(int, center)

            # 범위 체크
            if 0 <= x < self.width and 0 <= y < self.height:
                # Gaussian 블러로 부드럽게 누적
                self.add_gaussian_blob(x, y, radius=20, intensity=1.0)

    def add_gaussian_blob(self, x, y, radius=20, intensity=1.0):
        \"\"\"특정 위치에 Gaussian 분포로 값 누적\"\"\"
        # 범위 설정
        x_min = max(0, x - radius)
        x_max = min(self.width, x + radius)
        y_min = max(0, y - radius)
        y_max = min(self.height, y + radius)

        # Gaussian 커널 생성
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                # 거리 계산
                dist = np.sqrt((j - x)**2 + (i - y)**2)

                # Gaussian 함수
                if dist <= radius:
                    value = intensity * np.exp(-(dist**2) / (2 * (radius/3)**2))
                    self.heatmap[i, j] += value

    def get_heatmap_overlay(self, frame, alpha=0.5, colormap=cv2.COLORMAP_JET):
        \"\"\"
        히트맵 오버레이 생성

        Parameters:
            frame: 원본 프레임
            alpha: 투명도 (0-1)
            colormap: OpenCV 컬러맵
        \"\"\"
        # 1. 정규화 (0-255)
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255,
                                          cv2.NORM_MINMAX).astype(np.uint8)

        # 2. 컬러맵 적용
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)

        # 3. 원본 프레임과 합성
        overlay = cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)

        return overlay

    def get_hotspots(self, threshold_percentile=90):
        \"\"\"
        핫스팟 (활동 빈도 높은 영역) 추출

        Parameters:
            threshold_percentile: 상위 N% 영역을 핫스팟으로 간주

        Returns:
            List of hotspot regions (x, y, intensity)
        \"\"\"
        # 임계값 계산
        threshold = np.percentile(self.heatmap, threshold_percentile)

        # 핫스팟 영역 추출
        hotspot_mask = (self.heatmap > threshold).astype(np.uint8)

        # 연결 컴포넌트 분석
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            hotspot_mask, connectivity=8
        )

        hotspots = []
        for i in range(1, num_labels):  # 0은 배경
            x, y = centroids[i]
            intensity = self.heatmap[int(y), int(x)]
            area = stats[i, cv2.CC_STAT_AREA]

            hotspots.append({
                'center': (int(x), int(y)),
                'intensity': float(intensity),
                'area': int(area)
            })

        # 강도 순 정렬
        hotspots.sort(key=lambda h: h['intensity'], reverse=True)

        return hotspots

# 사용 예제
heatmap_gen = HeatmapGenerator(frame_shape=(720, 1280), decay_factor=0.995)

for frame in video_frames:
    # YOLOv8 + ByteTrack
    detections = yolo_model(frame)
    tracks = tracker.update(detections)

    # 히트맵 업데이트
    heatmap_gen.update(tracks)

    # 히트맵 오버레이
    overlay = heatmap_gen.get_heatmap_overlay(frame, alpha=0.6)

    # 핫스팟 추출 및 표시
    hotspots = heatmap_gen.get_hotspots(threshold_percentile=95)
    for idx, hotspot in enumerate(hotspots[:5]):  # 상위 5개
        x, y = hotspot['center']
        cv2.circle(overlay, (x, y), 30, (255, 255, 255), 2)
        cv2.putText(overlay, f"#{idx+1}", (x-10, y+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Heatmap', overlay)
        """, language="python")

        st.info("💡 **교육 포인트**: 히트맵은 시간에 따라 감쇠(decay)되어 최근 활동 강조")

        # 히트맵 시각화 옵션
        st.markdown("---")
        st.subheader("2️⃣ 히트맵 시각화 옵션")

        st.markdown("""
        #### 🎨 OpenCV 컬러맵

        OpenCV는 다양한 컬러맵을 제공합니다:
        """)

        st.code("""
# 주요 컬러맵
colormaps = {
    'JET': cv2.COLORMAP_JET,           # 🔵🟢🟡🔴 (기본, 가장 직관적)
    'HOT': cv2.COLORMAP_HOT,           # ⚫🔴🟡⚪ (열화상 카메라 스타일)
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,   # 🟣🔵🟢🟡 (과학적, 색맹 친화적)
    'TURBO': cv2.COLORMAP_TURBO,       # 🔵🟢🟡🔴 (JET 개선 버전)
    'RAINBOW': cv2.COLORMAP_RAINBOW,   # 🟣🔵🟢🟡🟠🔴 (무지개)
    'BONE': cv2.COLORMAP_BONE,         # ⚫⚪ (X-ray 스타일)
}

# 적용
heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        """, language="python")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📊 통계 분석

            ```python
            def analyze_heatmap(heatmap):
                \"\"\"히트맵 통계 분석\"\"\"

                # 기본 통계
                stats = {
                    'mean': np.mean(heatmap),
                    'median': np.median(heatmap),
                    'max': np.max(heatmap),
                    'std': np.std(heatmap)
                }

                # 활동 분포
                hist, bins = np.histogram(
                    heatmap.flatten(),
                    bins=50
                )

                # 핫스팟 비율
                threshold = np.percentile(heatmap, 90)
                hotspot_ratio = np.sum(heatmap > threshold) / heatmap.size

                stats['hotspot_ratio'] = hotspot_ratio

                return stats
            ```
            """)

        with col2:
            st.markdown("""
            #### 🕐 시간대별 분석

            ```python
            class TimeBasedHeatmap:
                def __init__(self, frame_shape):
                    self.heatmaps = defaultdict(
                        lambda: np.zeros(frame_shape, dtype=np.float32)
                    )

                def update(self, tracks, hour):
                    \"\"\"시간대별로 히트맵 분리\"\"\"
                    for track in tracks.values():
                        center = track.get_center(track.bbox)
                        x, y = map(int, center)

                        # 해당 시간대 히트맵에 누적
                        self.heatmaps[hour][y, x] += 1

                def get_peak_hours(self):
                    \"\"\"가장 활동이 많은 시간대\"\"\"
                    hour_activity = {
                        hour: np.sum(heatmap)
                        for hour, heatmap in self.heatmaps.items()
                    }
                    return sorted(hour_activity.items(),
                                 key=lambda x: x[1],
                                 reverse=True)
            ```
            """)

    def render_dashboard(self):
        """Tab 5: 간단한 대시보드"""
        st.header("📊 대시보드 (Dashboard)")

        st.markdown("---")

        # 실시간 통계
        st.subheader("1️⃣ 실시간 통계")

        st.markdown("""
        #### 📈 주요 지표 (KPI)

        CCTV 시스템에서 추적할 주요 지표:
        - 현재 탐지 객체 수
        - 총 추적 ID 수
        - 이벤트 발생 횟수 (침입/배회)
        - 평균 FPS
        - ROI 침입 횟수
        """)

        st.code("""
# 대시보드 메트릭 수집
class DashboardMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        \"\"\"메트릭 초기화\"\"\"
        self.total_detections = 0
        self.total_tracks = 0
        self.intrusion_events = 0
        self.loitering_events = 0
        self.fps_history = deque(maxlen=30)
        self.track_class_counts = defaultdict(int)  # {class_id: count}
        self.hourly_activity = defaultdict(int)  # {hour: count}

    def update(self, detections, tracks, events, fps, current_hour):
        \"\"\"프레임별 메트릭 업데이트\"\"\"

        # 탐지/추적 수
        self.total_detections += len(detections)
        self.total_tracks = len(tracks)

        # FPS
        self.fps_history.append(fps)

        # 클래스별 카운트
        for det in detections:
            self.track_class_counts[det['class']] += 1

        # 이벤트 카운트
        for event in events:
            if event['type'] == 'INTRUSION':
                self.intrusion_events += 1
            elif event['type'] == 'LOITERING':
                self.loitering_events += 1

        # 시간대별 활동
        self.hourly_activity[current_hour] += len(detections)

    def get_summary(self):
        \"\"\"요약 통계\"\"\"
        return {
            'current_tracks': self.total_tracks,
            'total_detections': self.total_detections,
            'intrusion_count': self.intrusion_events,
            'loitering_count': self.loitering_events,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'class_distribution': dict(self.track_class_counts),
            'peak_hour': max(self.hourly_activity.items(),
                           key=lambda x: x[1])[0] if self.hourly_activity else None
        }

# Streamlit 대시보드 예제
def render_dashboard(metrics):
    st.title("🎥 Smart CCTV Dashboard")

    summary = metrics.get_summary()

    # 1행: 주요 지표
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("현재 추적 중", summary['current_tracks'])

    with col2:
        st.metric("총 탐지 수", summary['total_detections'])

    with col3:
        st.metric("침입 이벤트", summary['intrusion_count'],
                 delta="+" + str(summary['intrusion_count']) if summary['intrusion_count'] > 0 else None)

    with col4:
        st.metric("평균 FPS", f"{summary['avg_fps']:.1f}")

    # 2행: 차트
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("클래스별 탐지 분포")
        if summary['class_distribution']:
            class_names = {0: 'Person', 2: 'Car', 3: 'Motorcycle'}
            chart_data = {
                class_names.get(k, f'Class {k}'): v
                for k, v in summary['class_distribution'].items()
            }
            st.bar_chart(chart_data)

    with col2:
        st.subheader("시간대별 활동")
        hourly_data = pd.DataFrame({
            'Hour': list(metrics.hourly_activity.keys()),
            'Activity': list(metrics.hourly_activity.values())
        })
        st.line_chart(hourly_data.set_index('Hour'))
        """, language="python")

        st.info("💡 **교육 포인트**: 실시간 대시보드로 시스템 상태 모니터링")

        # 이벤트 로그
        st.markdown("---")
        st.subheader("2️⃣ 이벤트 로그 (CSV)")

        st.markdown("""
        #### 📝 로그 시스템

        간소화 버전에서는 CSV 파일로 이벤트 기록:
        """)

        st.code("""
# CSV 로깅 시스템
import csv
from datetime import datetime
from pathlib import Path

class EventLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 날짜별 로그 파일
        today = datetime.now().strftime('%Y%m%d')
        self.log_file = self.log_dir / f'events_{today}.csv'

        # 헤더 작성 (파일이 없을 경우)
        if not self.log_file.exists():
            self.write_header()

    def write_header(self):
        \"\"\"CSV 헤더 작성\"\"\"
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Event Type',
                'Track ID',
                'Position X',
                'Position Y',
                'Duration',
                'Confidence',
                'Class',
                'Message'
            ])

    def log_event(self, event):
        \"\"\"이벤트 기록\"\"\"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                event.get('type', 'UNKNOWN'),
                event.get('track_id', ''),
                event.get('position', [0, 0])[0],
                event.get('position', [0, 0])[1],
                event.get('duration', 0),
                event.get('confidence', 0),
                event.get('class', ''),
                event.get('message', '')
            ])

    def read_logs(self, limit=100):
        \"\"\"최근 로그 읽기\"\"\"
        if not self.log_file.exists():
            return []

        with open(self.log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            logs = list(reader)

        # 최근 N개만 반환
        return logs[-limit:]

# 사용 예제
logger = EventLogger(log_dir='logs')

# 이벤트 발생 시
for event in alerts:
    logger.log_event(event)
    print(f"[LOG] {event['message']}")

# Streamlit에서 로그 표시
st.subheader("Recent Events")
logs = logger.read_logs(limit=50)
df = pd.DataFrame(logs)
st.dataframe(df, use_container_width=True)
        """, language="python")

        st.success("✅ **간소화**: CSV 로그 → 프로덕션에서는 PostgreSQL/MongoDB 사용")

        # 알림 시스템
        st.markdown("---")
        st.subheader("3️⃣ 알림 시스템 (교육용)")

        st.markdown("""
        #### 🔔 알림 방식

        **교육용 간소화**:
        - ✅ 콘솔 출력 (`print`)
        - ✅ Streamlit 토스트 알림
        - ❌ 이메일 (SMTP 설정 필요)
        - ❌ SMS (Twilio 계정 필요)
        - ❌ Webhook (서버 필요)
        """)

        st.code("""
# 간단한 알림 시스템
class SimpleAlertSystem:
    def __init__(self):
        self.alert_history = deque(maxlen=100)

    def send_alert(self, event):
        \"\"\"알림 발송\"\"\"

        # 1. 콘솔 출력
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [ALERT] {event['type']}: {event['message']}")

        # 2. 히스토리 저장
        event['timestamp'] = timestamp
        self.alert_history.append(event)

        # 3. Streamlit 토스트 (세션 상태 사용)
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        st.session_state.alerts.append(event)

    def get_recent_alerts(self, limit=10):
        \"\"\"최근 알림 조회\"\"\"
        return list(self.alert_history)[-limit:]

# Streamlit에서 알림 표시
def show_alerts():
    if 'alerts' in st.session_state and st.session_state.alerts:
        st.subheader("🚨 Recent Alerts")

        for alert in st.session_state.alerts[-5:]:  # 최근 5개
            if alert['type'] == 'INTRUSION':
                st.error(f"{alert['timestamp']} - {alert['message']}")
            elif alert['type'] == 'LOITERING':
                st.warning(f"{alert['timestamp']} - {alert['message']}")
            else:
                st.info(f"{alert['timestamp']} - {alert['message']}")

        # 초기화 버튼
        if st.button("Clear Alerts"):
            st.session_state.alerts = []
            st.rerun()
        """, language="python")

        # 통합 예제
        st.markdown("---")
        st.subheader("4️⃣ 통합 실행 예제")

        st.code("""
# 전체 시스템 통합
import cv2
from ultralytics import YOLO

def main():
    # 초기화
    model = YOLO('yolov8n.pt')
    tracker = SimpleByteTrack()

    # ROI 설정
    roi = np.array([[100, 200], [500, 200], [500, 400], [100, 400]])
    intrusion_detector = IntrusionDetector(roi, alert_threshold_seconds=3)
    loitering_detector = LoiteringDetector(min_duration_seconds=10, max_movement_pixels=100)

    # 히트맵
    heatmap_gen = HeatmapGenerator(frame_shape=(720, 1280))

    # 대시보드 & 로깅
    metrics = DashboardMetrics()
    logger = EventLogger()
    alert_system = SimpleAlertSystem()

    # 비디오 처리
    cap = cv2.VideoCapture('surveillance.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # 1. YOLOv8 탐지
        results = model(frame, conf=0.3, classes=[0, 2])  # 사람, 차량

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(box.conf[0]),
                    'class': int(box.cls[0])
                })

        # 2. ByteTrack 추적
        tracks = tracker.update(detections)

        # 3. 이벤트 감지
        current_time = frame_idx / fps
        current_hour = datetime.now().hour

        intrusion_alerts = intrusion_detector.check_intrusion(tracks, current_time)
        loitering_alerts = loitering_detector.check_loitering(tracks, current_time, fps)

        all_alerts = intrusion_alerts + loitering_alerts

        # 4. 알림 및 로깅
        for alert in all_alerts:
            logger.log_event(alert)
            alert_system.send_alert(alert)

        # 5. 히트맵 업데이트
        heatmap_gen.update(tracks)

        # 6. 메트릭 업데이트
        actual_fps = 1.0 / (time.time() - start_time)
        metrics.update(detections, tracks, all_alerts, actual_fps, current_hour)

        # 7. 시각화
        # 탐지/추적 그리기
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 침입/배회 오버레이
        frame = intrusion_detector.draw_intrusion_overlay(frame, tracks)
        frame = loitering_detector.draw_loitering_overlay(frame, tracks)

        # 히트맵 오버레이
        heatmap_overlay = heatmap_gen.get_heatmap_overlay(frame, alpha=0.4)

        # 정보 표시
        cv2.putText(frame, f"FPS: {actual_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 표시
        cv2.imshow('Smart CCTV - Main', frame)
        cv2.imshow('Smart CCTV - Heatmap', heatmap_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # 종료
    cap.release()
    cv2.destroyAllWindows()

    # 최종 요약
    summary = metrics.get_summary()
    print("\\n=== Final Summary ===")
    print(f"Total Detections: {summary['total_detections']}")
    print(f"Intrusion Events: {summary['intrusion_count']}")
    print(f"Loitering Events: {summary['loitering_count']}")
    print(f"Average FPS: {summary['avg_fps']:.1f}")

if __name__ == '__main__':
    main()
        """, language="python")

        st.success("""
        ✅ **교육용 시스템 완성**:
        - YOLOv8 + ByteTrack 통합
        - ROI 침입/배회 감지
        - 히트맵 분석
        - 간단한 로깅 및 알림
        - 실시간 대시보드
        """)


def main():
    """Streamlit 앱 진입점"""
    module = SmartCCTVModule()
    module.render()


if __name__ == "__main__":
    main()
