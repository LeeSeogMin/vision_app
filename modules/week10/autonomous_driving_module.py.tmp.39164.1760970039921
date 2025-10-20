"""
Week 10: 자율주행 인식 시스템 (End-to-End Autonomous Driving Pipeline)
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import io
import os

from core.base_processor import BaseImageProcessor


class AutonomousDrivingModule(BaseImageProcessor):
    """Week 10: 자율주행 인식 시스템 모듈"""

    def __init__(self):
        super().__init__()
        self.name = "Week 10: Autonomous Driving Pipeline"

    def render(self):
        """메인 렌더링 함수"""
        st.title("🚗 Week 10: 자율주행 인식 시스템")

        st.markdown("""
        ## 학습 목표
        - **이론**: SAE 자율주행 레벨, 센서 융합, 인식-판단-제어 파이프라인
        - **차선 인식**: 직선(Hough) → 곡선(Polynomial) → 딥러닝(LaneNet)
        - **객체 탐지**: YOLOv8 + ByteTrack + IPM 거리 추정
        - **통합 시스템**: 위험도 분석 + 의사결정 로직
        - **3D 시각화**: BEV(Bird's Eye View) + 3D 바운딩 박스
        - **실전**: 실시간 추론, TensorRT 최적화, Edge 배포
        """)

        # 환경 체크
        self._check_environment()

        # 7개 탭 구성
        tabs = st.tabs([
            "📚 자율주행 이론",
            "🛣️ 차선 인식 (3-Tier)",
            "🚙 객체 탐지 및 추적",
            "🔗 통합 파이프라인",
            "📐 3D 시각화 (BEV)",
            "🎮 고급 시뮬레이터",
            "💻 실전 배포"
        ])

        with tabs[0]:
            self.render_theory()

        with tabs[1]:
            self.render_lane_detection()

        with tabs[2]:
            self.render_object_detection()

        with tabs[3]:
            self.render_integrated_pipeline()

        with tabs[4]:
            self.render_3d_visualization()

        with tabs[5]:
            self.render_simulator()

        with tabs[6]:
            self.render_deployment()

    def _check_environment(self):
        """환경 체크 및 설정"""
        with st.expander("🔧 환경 설정 확인", expanded=False):
            st.markdown("""
            ### 필요한 패키지
            - `opencv-python`: 영상 처리, 차선 인식
            - `ultralytics`: YOLOv8 객체 탐지
            - `numpy`, `matplotlib`: 시뮬레이션 및 시각화
            - `torch`: 딥러닝 모델 (선택적)

            ### 3-Tier 실행 전략
            1. **Full Mode**: 모든 패키지 설치 (권장)
            2. **Basic Mode**: OpenCV + YOLO만 사용
            3. **Simulation Mode**: 시뮬레이션만 실행
            """)

            issues = []

            # Check opencv
            try:
                import cv2
                st.success(f"✅ opencv-python {cv2.__version__}")
            except ImportError:
                issues.append("opencv-python")
                st.warning("⚠️ opencv-python 미설치")

            # Check ultralytics
            try:
                import ultralytics
                st.success(f"✅ ultralytics (YOLOv8)")
            except ImportError:
                issues.append("ultralytics")
                st.warning("⚠️ ultralytics 미설치")

            # Check torch
            try:
                import torch
                device = "GPU" if torch.cuda.is_available() else "CPU"
                st.success(f"✅ torch ({device})")
            except ImportError:
                issues.append("torch")
                st.info("ℹ️ torch 미설치 (딥러닝 기능 제한)")

            if issues:
                st.info(f"""
                ### 🔧 설치 방법
                ```bash
                pip install opencv-python ultralytics torch matplotlib numpy
                ```
                """)

    # ==================== Tab 1: 자율주행 이론 ====================

    def render_theory(self):
        """자율주행 이론 설명"""
        st.header("📚 자율주행 이론 및 시스템 구조")

        # 1. SAE 자율주행 레벨
        st.markdown("""
        ## 1. SAE 자율주행 레벨 (SAE J3016)

        SAE(Society of Automotive Engineers)에서 정의한 자율주행 레벨은 자동화 수준에 따라 0~5단계로 구분됩니다.
        """)

        level_cols = st.columns(6)
        levels = [
            ("레벨 0", "완전 수동", "운전자가 모든 제어", "❌", "#FF4444"),
            ("레벨 1", "운전자 보조", "ACC, LKA 등", "⚡", "#FF8844"),
            ("레벨 2", "부분 자동화", "조향+가감속", "🔄", "#FFBB44"),
            ("레벨 3", "조건부 자동화", "특정 조건 자율", "🚗", "#88DD44"),
            ("레벨 4", "고도 자동화", "대부분 자율", "🤖", "#44BBFF"),
            ("레벨 5", "완전 자동화", "모든 상황 자율", "✨", "#8844FF")
        ]

        for col, (level, name, desc, icon, color) in zip(level_cols, levels):
            with col:
                st.markdown(f"""
                <div style="background:{color}; padding:15px; border-radius:10px; text-align:center;">
                    <h1>{icon}</h1>
                    <h4>{level}</h4>
                    <p style="font-size:12px;"><b>{name}</b></p>
                    <p style="font-size:10px;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 2. 자율주행 시스템 구조
        st.markdown("""
        ## 2. 자율주행 시스템 3단계 구조

        자율주행은 **인식(Perception) → 판단(Planning) → 제어(Control)** 3단계로 구성됩니다.
        """)

        with st.expander("📸 1단계: 인식 (Perception)", expanded=True):
            st.markdown("""
            **목적**: 주변 환경을 이해하고 디지털 정보로 변환

            **센서 종류**:
            - **카메라**: 색상, 형태, 텍스트 인식 (신호등, 표지판)
            - **라이다(LiDAR)**: 3D 거리 측정, 정밀한 물체 위치
            - **레이더(Radar)**: 장거리 속도 측정, 악천후 강건
            - **초음파**: 근거리 장애물 감지 (주차 보조)

            **인식 기술**:
            - 차선 인식 (Lane Detection)
            - 객체 탐지 (Object Detection): 차량, 보행자, 신호등
            - 객체 추적 (Object Tracking): ID 유지
            - 거리 추정 (Depth Estimation)
            - 세그멘테이션 (Semantic/Instance Segmentation)

            **센서 융합 (Sensor Fusion)**:
            - 카메라 + 라이다 → 정확한 3D 위치
            - 레이더 + 카메라 → 악천후 대응
            - 칼만 필터, 베이지안 융합
            """)

        with st.expander("🧠 2단계: 판단 (Planning)", expanded=True):
            st.markdown("""
            **목적**: 인식 정보를 바탕으로 주행 전략 수립

            **3-Level 계획**:

            1. **Mission Planning (미션 계획)**
               - 목적지까지 전체 경로 계획
               - 고속도로 vs 일반도로 선택
               - 휴게소, 충전소 경유지

            2. **Behavioral Planning (행동 계획)**
               - 차선 변경, 추월, 회전 결정
               - 신호등 대기, 보행자 양보
               - 교차로 진입 타이밍

            3. **Motion Planning (동작 계획)**
               - 최적 궤적(Trajectory) 생성
               - 가속/감속 프로파일
               - 장애물 회피 경로

            **위험도 분석**:
            - TTC (Time To Collision): 충돌까지 남은 시간
            - 차선 이탈 위험도
            - 사각지대 경고

            **의사결정 우선순위**:
            1. 안전 (Safety First)
            2. 법규 준수 (Traffic Rules)
            3. 승차감 (Comfort)
            4. 효율성 (Efficiency)
            """)

        with st.expander("🎮 3단계: 제어 (Control)", expanded=True):
            st.markdown("""
            **목적**: 계획된 경로를 정확히 추종하도록 차량 제어

            **제어 알고리즘**:
            - **PID Controller**: 비례-적분-미분 제어
            - **MPC (Model Predictive Control)**: 모델 기반 예측 제어
            - **Pure Pursuit**: 경로 추종 알고리즘
            - **Stanley Controller**: 횡방향 제어

            **제어 대상**:
            - 조향각 (Steering Angle)
            - 가속/감속 (Throttle/Brake)
            - 기어 변속

            **피드백 루프**:
            ```
            계획 경로 → 제어기 → 액추에이터 → 차량 동역학
                ↑                                      ↓
                └───────── 센서 피드백 ←──────────────┘
            ```
            """)

        st.markdown("---")

        # 3. Week 10 파이프라인
        st.markdown("""
        ## 3. Week 10 End-to-End 파이프라인

        이번 주차에서는 **인식(Perception)** 단계를 집중적으로 다룹니다.
        """)

        st.code("""
# Week 10 파이프라인 구조

입력: 도로 영상 (Video Stream)
    ↓
[1단계] 차선 인식 (Lane Detection)
    ├─ Tier 1: Hough Transform (직선)
    ├─ Tier 2: Polynomial Fitting (곡선)
    └─ Tier 3: LaneNet (딥러닝)
    ↓
[2단계] 객체 탐지 (Object Detection)
    ├─ YOLOv8: 차량/보행자/신호등 탐지
    ├─ ByteTrack: ID 유지 추적
    └─ IPM: 거리 추정
    ↓
[3단계] 위험도 분석 (Risk Analysis)
    ├─ 차선 이탈 위험도 (0~1)
    ├─ 충돌 위험도 (TTC)
    └─ 급정거 차량 감지
    ↓
[4단계] 의사결정 (Decision Making)
    ├─ 차선 복귀 (STEER_BACK)
    ├─ 긴급 제동 (EMERGENCY_BRAKE)
    └─ 감속 (SLOW_DOWN)
    ↓
[5단계] 시각화 (Visualization)
    ├─ 2D: 바운딩 박스, 차선 오버레이
    └─ 3D: BEV (Bird's Eye View)
        """, language='text')

        st.markdown("---")

        # 4. 실제 사례
        st.markdown("""
        ## 4. 실제 자율주행 시스템 비교
        """)

        comparison_data = {
            "시스템": ["Tesla Autopilot", "Waymo Driver", "GM Cruise", "현대 Highway"],
            "레벨": ["2-3", "4", "4", "2"],
            "센서": ["8 카메라", "카메라+라이다+레이더", "카메라+라이다", "카메라+레이더"],
            "운행 지역": ["전세계", "미국 일부", "샌프란시스코", "고속도로"],
            "특징": ["카메라 중심", "라이다 의존", "도심 주행", "국내 최초 L3"]
        }

        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # 5. 과제
        st.markdown("""
        ## 5. 학습 과제

        **이론 학습**:
        - [ ] SAE 레벨별 차이점 이해
        - [ ] 센서 융합의 필요성 이해
        - [ ] 인식-판단-제어 파이프라인 그려보기

        **실습 준비**:
        - [ ] OpenCV, YOLO 환경 설정
        - [ ] 샘플 도로 영상 준비 (road_video.mp4)
        - [ ] 다음 탭에서 차선 인식 실습 진행
        """)

    # ==================== Tab 2: 차선 인식 ====================

    def render_lane_detection(self):
        """차선 인식 3-Tier 구현"""
        st.header("🛣️ 차선 인식 (Lane Detection)")

        st.markdown("""
        ## 차선 인식의 중요성

        차선 인식은 자율주행의 **기본**이자 **핵심**입니다:
        - 차량의 현재 차선 위치 파악
        - 차선 이탈 경고 (LDWS)
        - 차선 유지 보조 (LKAS)
        - 주행 가능 영역 정의

        ### 3-Tier 접근법

        난이도와 정확도에 따라 3가지 방법을 학습합니다.
        """)

        tier_tabs = st.tabs(["Tier 1: 직선 차선", "Tier 2: 곡선 차선", "Tier 3: 딥러닝", "📊 비교 분석"])

        with tier_tabs[0]:
            self._render_lane_tier1()

        with tier_tabs[1]:
            self._render_lane_tier2()

        with tier_tabs[2]:
            self._render_lane_tier3()

        with tier_tabs[3]:
            self._render_lane_comparison()

    def _render_lane_tier1(self):
        """Tier 1: Hough Transform 직선 차선"""
        st.subheader("Tier 1: Hough Transform (직선 차선)")

        st.markdown("""
        **개념**: 전통적인 컴퓨터 비전 기법으로 직선 차선을 검출합니다.

        **장점**: 빠름, 간단함, 실시간 처리 가능
        **단점**: 곡선 차선 처리 불가, 악천후 약함
        **적용**: 고속도로 직선 구간
        """)

        # 5단계 파이프라인
        pipeline_cols = st.columns(5)
        steps = [
            ("1️⃣", "전처리", "Gray + Blur"),
            ("2️⃣", "엣지 검출", "Canny Edge"),
            ("3️⃣", "ROI 설정", "관심 영역"),
            ("4️⃣", "직선 검출", "Hough Transform"),
            ("5️⃣", "시각화", "Overlay")
        ]

        for col, (icon, title, desc) in zip(pipeline_cols, steps):
            with col:
                st.markdown(f"""
                <div style="background:#f0f2f6; padding:10px; border-radius:5px; text-align:center;">
                    <h2>{icon}</h2>
                    <h5>{title}</h5>
                    <p style="font-size:11px;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 상세 설명
        with st.expander("🔍 각 단계 상세 설명", expanded=True):
            st.markdown("""
            ### 1단계: 전처리 (Preprocessing)

            ```python
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 가우시안 블러 (노이즈 제거)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            ```

            **이유**: 색상 정보가 필요 없고, 블러로 노이즈를 줄여 엣지 검출 정확도 향상

            ---

            ### 2단계: 캐니 엣지 검출 (Canny Edge Detection)

            ```python
            edges = cv2.Canny(blur, low_threshold=50, high_threshold=150)
            ```

            **원리**:
            1. Gradient 계산 (밝기 변화)
            2. Non-Maximum Suppression (가장 강한 엣지만)
            3. Double Threshold (약한/강한 엣지 구분)
            4. Edge Tracking (연결된 엣지만 유지)

            **파라미터**:
            - `low_threshold=50`: 약한 엣지 임계값
            - `high_threshold=150`: 강한 엣지 임계값

            ---

            ### 3단계: ROI (Region of Interest) 설정

            ```python
            height, width = frame.shape[:2]

            # 삼각형 ROI (차선이 있을 영역)
            roi_vertices = np.array([[
                (0, height),                  # 좌하단
                (width/2, height/2),          # 상단 중앙
                (width, height)               # 우하단
            ]], dtype=np.int32)

            # 마스크 생성
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, roi_vertices, 255)

            # 마스크 적용
            masked_edges = cv2.bitwise_and(edges, mask)
            ```

            **이유**: 하늘, 나무, 간판 등 불필요한 엣지 제거 → 처리 속도 향상

            ---

            ### 4단계: Hough Transform 직선 검출

            ```python
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=2,              # 거리 해상도 (픽셀)
                theta=np.pi/180,    # 각도 해상도 (1도)
                threshold=50,       # 최소 교차점 수
                minLineLength=40,   # 최소 선 길이
                maxLineGap=100      # 최대 선 간격
            )
            ```

            **Hough Transform 원리**:
            - 이미지 공간(x, y) → Hough 공간(ρ, θ) 변환
            - 직선: y = mx + c → ρ = x·cos(θ) + y·sin(θ)
            - 많은 점이 교차하는 (ρ, θ)가 직선

            **파라미터 튜닝**:
            - `threshold` ↑ → 긴 직선만 검출
            - `minLineLength` ↑ → 짧은 선 제거
            - `maxLineGap` ↑ → 끊어진 선 연결

            ---

            ### 5단계: 시각화

            ```python
            line_image = np.zeros_like(frame)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 반투명 합성
            result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
            ```
            """)

        st.markdown("---")

        # 전체 코드
        with st.expander("📋 전체 Tier 1 코드 (Colab/로컬)", expanded=False):
            st.code("""
import cv2
import numpy as np

def detect_lanes_hough(frame):
    \"\"\"Tier 1: Hough Transform 차선 인식\"\"\"

    # 1. 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 캐니 엣지 검출
    edges = cv2.Canny(blur, 50, 150)

    # 3. ROI 설정
    height, width = frame.shape[:2]
    roi_vertices = np.array([[
        (0, height),
        (width/2, height/2),
        (width, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # 4. Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi/180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    # 5. 시각화
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return result


def main():
    # 비디오 로드
    cap = cv2.VideoCapture('road_video.mp4')  # 또는 0 (웹캠)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 차선 인식
        result = detect_lanes_hough(frame)

        # 결과 표시
        cv2.imshow('Tier 1: Hough Transform', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
            """, language='python')

        st.markdown("---")

        # 파라미터 실험
        st.markdown("### 🧪 파라미터 실험 (시뮬레이션)")

        col1, col2 = st.columns(2)
        with col1:
            canny_low = st.slider("Canny Low Threshold", 0, 100, 50)
            canny_high = st.slider("Canny High Threshold", 100, 300, 150)
        with col2:
            hough_threshold = st.slider("Hough Threshold", 10, 100, 50)
            min_line_length = st.slider("Min Line Length", 10, 100, 40)

        st.info(f"""
        **현재 설정**:
        - Canny: [{canny_low}, {canny_high}]
        - Hough Threshold: {hough_threshold}
        - Min Line Length: {min_line_length}

        💡 **팁**: Canny를 낮추면 더 많은 엣지, Hough를 높이면 더 확실한 직선만 검출
        """)

    def _render_lane_tier2(self):
        """Tier 2: Polynomial Fitting 곡선 차선"""
        st.subheader("Tier 2: Polynomial Fitting (곡선 차선)")

        st.markdown("""
        **개념**: Sliding Window와 다항식 피팅으로 곡선 차선을 검출합니다.

        **장점**: 곡선 도로 처리 가능, 차선 곡률 계산 가능
        **단점**: Tier 1보다 느림, 파라미터 튜닝 필요
        **적용**: 일반 도로, 커브 구간
        """)

        # 알고리즘 단계
        st.markdown("### 알고리즘 단계")

        step_cols = st.columns(4)
        steps = [
            ("1️⃣", "Perspective\nTransform", "BEV 변환"),
            ("2️⃣", "Histogram\nPeak", "차선 시작점"),
            ("3️⃣", "Sliding\nWindow", "차선 픽셀 추출"),
            ("4️⃣", "Polynomial\nFit", "2차 곡선 피팅")
        ]

        for col, (icon, title, desc) in zip(step_cols, steps):
            with col:
                st.markdown(f"""
                <div style="background:#e8f4f8; padding:12px; border-radius:5px; text-align:center;">
                    <h2>{icon}</h2>
                    <p style="font-size:13px; margin:0;"><b>{title}</b></p>
                    <p style="font-size:11px; color:#666;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        with st.expander("🔍 Perspective Transform (BEV 변환)", expanded=True):
            st.markdown("""
            **목적**: 차선을 위에서 본 것처럼 변환 (Bird's Eye View)

            ```python
            # Source points (원본 4점)
            src = np.float32([
                [width * 0.2, height],        # 좌하단
                [width * 0.45, height * 0.6], # 좌상단
                [width * 0.55, height * 0.6], # 우상단
                [width * 0.8, height]         # 우하단
            ])

            # Destination points (변환 후 4점)
            dst = np.float32([
                [width * 0.2, height],        # 좌하단
                [width * 0.2, 0],             # 좌상단
                [width * 0.8, 0],             # 우상단
                [width * 0.8, height]         # 우하단
            ])

            # 변환 행렬 계산
            M = cv2.getPerspectiveTransform(src, dst)

            # 변환 적용
            warped = cv2.warpPerspective(edges, M, (width, height))
            ```

            **효과**: 평행하지 않은 차선이 평행하게 보임 → 다항식 피팅 용이
            """)

        with st.expander("🔍 Histogram & Sliding Window", expanded=True):
            st.markdown("""
            ### Histogram으로 차선 시작점 찾기

            ```python
            # 이미지 하단 절반의 히스토그램
            histogram = np.sum(warped[height//2:, :], axis=0)

            # 좌우 차선의 시작점 (피크 위치)
            midpoint = len(histogram) // 2
            left_base = np.argmax(histogram[:midpoint])
            right_base = np.argmax(histogram[midpoint:]) + midpoint
            ```

            ### Sliding Window로 차선 픽셀 추출

            ```python
            # 윈도우 개수 (9개)
            nwindows = 9
            window_height = height // nwindows

            # 좌우 차선 픽셀 저장
            left_lane_inds = []
            right_lane_inds = []

            # 각 윈도우마다 반복
            for window in range(nwindows):
                # 윈도우 경계 계산
                win_y_low = height - (window + 1) * window_height
                win_y_high = height - window * window_height

                # 좌측 윈도우
                win_xleft_low = left_current - margin
                win_xleft_high = left_current + margin

                # 윈도우 내 흰색 픽셀 찾기
                good_left_inds = ((nonzeroy >= win_y_low) &
                                  (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) &
                                  (nonzerox < win_xleft_high)).nonzero()[0]

                left_lane_inds.append(good_left_inds)

                # 충분한 픽셀이 있으면 중심 업데이트
                if len(good_left_inds) > minpix:
                    left_current = int(np.mean(nonzerox[good_left_inds]))
            ```

            **윈도우 파라미터**:
            - `nwindows=9`: 윈도우 개수
            - `margin=100`: 윈도우 폭의 절반
            - `minpix=50`: 중심 업데이트 최소 픽셀 수
            """)

        with st.expander("🔍 Polynomial Fitting (다항식 피팅)", expanded=True):
            st.markdown("""
            ### 2차 다항식으로 차선 곡선 피팅

            ```python
            # 추출된 차선 픽셀
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]

            # 2차 다항식 피팅: x = ay² + by + c
            left_fit = np.polyfit(lefty, leftx, 2)

            # 곡선 생성
            ploty = np.linspace(0, height-1, height)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            ```

            **왜 2차식인가?**:
            - 대부분의 도로 곡선은 2차 함수로 근사 가능
            - 계산 효율적
            - 과적합(Overfitting) 방지

            ### 차선 곡률 계산

            ```python
            # 미터 단위 변환
            ym_per_pix = 30/720  # y축: 30미터 / 720픽셀
            xm_per_pix = 3.7/700 # x축: 3.7미터 / 700픽셀

            # 실제 좌표로 다시 피팅
            left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)

            # 곡률 반지름 계산
            y_eval = np.max(ploty)
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix +
                                   left_fit_cr[1])**2)**1.5) / np.abs(2*left_fit_cr[0])

            print(f"차선 곡률: {left_curverad:.0f} 미터")
            ```

            **활용**:
            - 곡률 → 조향각 계산
            - 급커브 경고
            - 속도 제한 권장
            """)

        st.markdown("---")

        # 전체 코드
        with st.expander("📋 전체 Tier 2 코드 (Colab/로컬)", expanded=False):
            st.code("""
import cv2
import numpy as np

def detect_lanes_polynomial(frame):
    \"\"\"Tier 2: Polynomial Fitting 차선 인식\"\"\"

    # 1. 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2. Perspective Transform (BEV)
    height, width = frame.shape[:2]
    src = np.float32([
        [width * 0.2, height],
        [width * 0.45, height * 0.6],
        [width * 0.55, height * 0.6],
        [width * 0.8, height]
    ])
    dst = np.float32([
        [width * 0.2, height],
        [width * 0.2, 0],
        [width * 0.8, 0],
        [width * 0.8, height]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(edges, M, (width, height))

    # 3. Histogram으로 시작점 찾기
    histogram = np.sum(warped[height//2:, :], axis=0)
    midpoint = len(histogram) // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # 4. Sliding Window
    nwindows = 9
    window_height = height // nwindows
    margin = 100
    minpix = 50

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_current = left_base
    right_current = right_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height

        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            left_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            right_current = int(np.mean(nonzerox[good_right_inds]))

    # 5. Polynomial Fitting
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 6. 곡선 생성
    ploty = np.linspace(0, height-1, height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # 7. 시각화
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Inverse Perspective Transform
    Minv = cv2.getPerspectiveTransform(dst, src)
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))

    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

    return result


def main():
    cap = cv2.VideoCapture('road_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_lanes_polynomial(frame)
        cv2.imshow('Tier 2: Polynomial Fitting', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
            """, language='python')

    def _render_lane_tier3(self):
        """Tier 3: 딥러닝 차선 인식"""
        st.subheader("Tier 3: 딥러닝 (LaneNet, SCNN)")

        st.markdown("""
        **개념**: 딥러닝 모델로 차선을 세그멘테이션합니다.

        **장점**:
        - 복잡한 환경 대응 (악천후, 야간, 그림자)
        - 곡선/직선 구분 불필요
        - End-to-End 학습 가능

        **단점**:
        - 느림 (GPU 필요)
        - 학습 데이터 필요
        - 모델 크기 큼

        **적용**: 모든 도로 환경
        """)

        # 딥러닝 모델 비교
        st.markdown("### 주요 딥러닝 모델 비교")

        model_data = {
            "모델": ["LaneNet", "SCNN", "Ultra-Fast-Lane", "PolyLaneNet"],
            "구조": ["Encoder-Decoder", "Slice CNN", "Row Anchor", "Polynomial"],
            "FPS": ["~30", "~15", "~320", "~80"],
            "정확도": ["높음", "매우 높음", "중간", "높음"],
            "특징": ["Instance Seg", "Spatial Info", "초고속", "곡선 적합"]
        }

        import pandas as pd
        df = pd.DataFrame(model_data)
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        with st.expander("🔍 LaneNet 아키텍처", expanded=True):
            st.markdown("""
            ### LaneNet 구조

            ```
            입력 이미지 (H × W × 3)
                ↓
            ┌─────────────────────────────┐
            │ Encoder (ENet)              │
            │  - Feature Extraction       │
            │  - Downsampling             │
            └─────────────────────────────┘
                ↓
            ┌────────────────┬────────────────┐
            │ Binary Branch  │ Embedding Br.  │
            │ (차선 여부)     │ (차선 ID)      │
            └────────────────┴────────────────┘
                ↓                   ↓
            Binary Seg Map    Instance Seg Map
            (H × W × 1)      (H × W × 4)
                ↓                   ↓
            ┌─────────────────────────────────┐
            │ Post-Processing                  │
            │  - Clustering (DBSCAN)          │
            │  - Curve Fitting                │
            └─────────────────────────────────┘
                ↓
            차선 좌표 리스트
            ```

            ### Binary Segmentation Branch

            - **목적**: 픽셀이 차선인지 아닌지 분류 (0 or 1)
            - **손실 함수**: Binary Cross-Entropy

            ### Embedding Branch

            - **목적**: 각 차선에 고유한 ID 부여 (Instance Segmentation)
            - **손실 함수**: Discriminative Loss
              - 같은 차선 픽셀끼리는 가깝게
              - 다른 차선 픽셀끼리는 멀게

            ### Post-Processing

            1. **Clustering**: 같은 임베딩 값을 가진 픽셀을 하나의 차선으로 그룹화
            2. **Curve Fitting**: 각 차선 픽셀에 다항식 피팅
            """)

        with st.expander("🔍 사전훈련 모델 사용하기", expanded=True):
            st.markdown("""
            ### HuggingFace에서 모델 로드

            ```python
            # Transformers 라이브러리로 Segformer 사용
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
            import torch
            from PIL import Image

            # 모델 및 전처리기 로드
            feature_extractor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )

            # 추론
            def detect_lanes_dl(image):
                # 전처리
                inputs = feature_extractor(images=image, return_tensors="pt")

                # 모델 예측
                with torch.no_grad():
                    outputs = model(**inputs)

                # Logits → Segmentation Map
                logits = outputs.logits
                seg_map = torch.argmax(logits, dim=1)[0]

                # 차선 클래스만 추출 (클래스 번호는 데이터셋 의존)
                lane_mask = (seg_map == LANE_CLASS_ID).numpy()

                return lane_mask
            ```

            ### TuSimple/CULane 데이터셋으로 파인튜닝

            ```python
            from torch.utils.data import DataLoader
            from transformers import Trainer, TrainingArguments

            # 학습 설정
            training_args = TrainingArguments(
                output_dir="./lane_model",
                per_device_train_batch_size=4,
                num_train_epochs=50,
                learning_rate=5e-5,
                logging_steps=100,
                save_steps=500,
            )

            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            # 학습 시작
            trainer.train()
            ```
            """)

        st.markdown("---")

        # 전체 코드 (간단 버전)
        with st.expander("📋 Tier 3 코드 (사전훈련 모델 사용)", expanded=False):
            st.code("""
# Tier 3: 딥러닝 차선 인식 (Segformer 사용)

import cv2
import numpy as np
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image

# 모델 로드 (한 번만)
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# GPU 사용 (가능하면)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


def detect_lanes_dl(frame):
    \"\"\"딥러닝 기반 차선 인식\"\"\"

    # OpenCV (BGR) → PIL (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 전처리
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 추론
    with torch.no_grad():
        outputs = model(**inputs)

    # Logits → Segmentation Map
    logits = outputs.logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False
    )
    seg_map = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # 차선 클래스 추출 (예: 클래스 6이 도로라고 가정)
    # 실제로는 TuSimple/CULane 파인튜닝 필요
    lane_mask = (seg_map == 6).astype(np.uint8) * 255

    # 색상 오버레이
    lane_color = np.zeros_like(frame)
    lane_color[lane_mask > 0] = [0, 255, 0]  # 초록색

    result = cv2.addWeighted(frame, 0.7, lane_color, 0.3, 0)

    return result


def main():
    cap = cv2.VideoCapture('road_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_lanes_dl(frame)
        cv2.imshow('Tier 3: Deep Learning', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# 참고: 실제 차선 인식을 위해서는 TuSimple/CULane 데이터셋으로
# 파인튜닝한 모델을 사용해야 합니다.
# HuggingFace Hub에서 "lane-detection" 검색하여 사전훈련 모델 찾기
            """, language='python')

    def _render_lane_comparison(self):
        """3-Tier 비교 분석"""
        st.subheader("📊 3-Tier 비교 분석")

        st.markdown("""
        ### 성능 비교표
        """)

        comparison_data = {
            "지표": ["처리 속도 (FPS)", "정확도 (%)", "곡선 대응", "악천후 강건성", "GPU 필요", "구현 난이도", "권장 환경"],
            "Tier 1 (Hough)": ["60-120", "75-85", "❌ 불가", "⚠️ 약함", "❌", "⭐", "고속도로 직선"],
            "Tier 2 (Polynomial)": ["30-60", "85-92", "✅ 가능", "⚠️ 약함", "❌", "⭐⭐⭐", "일반 도로 곡선"],
            "Tier 3 (Deep Learning)": ["10-30", "92-98", "✅ 가능", "✅ 강함", "✅", "⭐⭐⭐⭐⭐", "모든 환경"]
        }

        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        # 시각적 비교
        st.markdown("### 시각적 비교")

        scenario_cols = st.columns(3)
        scenarios = [
            ("🌞 맑은 날 직선", "Tier 1: ⭐⭐⭐⭐⭐\nTier 2: ⭐⭐⭐⭐\nTier 3: ⭐⭐⭐⭐⭐"),
            ("🌀 곡선 도로", "Tier 1: ⭐\nTier 2: ⭐⭐⭐⭐\nTier 3: ⭐⭐⭐⭐⭐"),
            ("🌧️ 비오는 날", "Tier 1: ⭐\nTier 2: ⭐⭐\nTier 3: ⭐⭐⭐⭐")
        ]

        for col, (title, ratings) in zip(scenario_cols, scenarios):
            with col:
                st.markdown(f"""
                <div style="background:#f8f9fa; padding:15px; border-radius:10px;">
                    <h4>{title}</h4>
                    <pre style="font-size:12px;">{ratings}</pre>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 선택 가이드
        st.markdown("""
        ### 🎯 선택 가이드

        **Tier 1 (Hough Transform)을 선택하세요:**
        - ✅ 고속도로 직선 구간
        - ✅ 실시간 처리 필수 (임베디드 시스템)
        - ✅ 간단한 프로토타입
        - ✅ GPU 없음

        **Tier 2 (Polynomial)를 선택하세요:**
        - ✅ 일반 도로 (곡선 포함)
        - ✅ 차선 곡률 정보 필요
        - ✅ 중간 정확도 요구
        - ✅ GPU 없음

        **Tier 3 (Deep Learning)을 선택하세요:**
        - ✅ 복잡한 환경 (도심, 공사구간)
        - ✅ 악천후 대응 필수
        - ✅ 최고 정확도 요구
        - ✅ GPU 사용 가능
        - ✅ 학습 데이터 확보 가능

        ---

        ### 💡 실전 팁: 하이브리드 접근

        실제 자율주행 시스템은 여러 방법을 조합합니다:

        ```python
        def detect_lanes_hybrid(frame):
            # 1차: 딥러닝 (신뢰도 높으면 바로 사용)
            lane_dl, confidence = detect_lanes_dl(frame)
            if confidence > 0.9:
                return lane_dl

            # 2차: Polynomial Fallback
            lane_poly = detect_lanes_polynomial(frame)
            if is_valid(lane_poly):
                return lane_poly

            # 3차: Hough Fallback
            return detect_lanes_hough(frame)
        ```

        **장점**:
        - 정상 상황: 딥러닝의 높은 정확도
        - 비정상 상황: 전통 방식의 안정성
        - 실패 확률 최소화
        """)

    # ==================== Tab 3: 객체 탐지 및 추적 ====================

    def render_object_detection(self):
        """객체 탐지 및 추적 구현"""
        st.header("🚙 객체 탐지 및 추적")

        st.markdown("""
        ## 객체 탐지의 중요성

        자율주행에서 주변 객체를 정확히 인식하는 것은 안전의 핵심입니다:
        - 차량: 충돌 방지, 차간 거리 유지
        - 보행자: 횡단 감지, 급정거
        - 신호등/표지판: 교통 규칙 준수
        - 기타: 이륜차, 동물, 낙하물
        """)

        obj_tabs = st.tabs([
            "🎯 YOLOv8 탐지",
            "🔗 ByteTrack 추적",
            "📏 거리 추정 (IPM)",
            "📋 전체 코드"
        ])

        with obj_tabs[0]:
            self._render_yolov8()

        with obj_tabs[1]:
            self._render_bytetrack()

        with obj_tabs[2]:
            self._render_ipm()

        with obj_tabs[3]:
            self._render_object_full_code()

    def _render_yolov8(self):
        """YOLOv8 객체 탐지"""
        st.subheader("🎯 YOLOv8 객체 탐지")

        st.markdown("""
        **YOLO (You Only Look Once)**: 실시간 객체 탐지의 대표 알고리즘

        **YOLOv8 특징**:
        - Anchor-Free 디자인
        - 빠른 속도 (V100 GPU에서 ~200 FPS)
        - 높은 정확도 (COCO mAP 53.9%)
        - 다양한 크기 (n/s/m/l/x)
        """)

        # YOLOv8 모델 크기 비교
        model_cols = st.columns(5)
        models = [
            ("YOLOv8n", "Nano", "3.2M", "~300 FPS", "임베디드"),
            ("YOLOv8s", "Small", "11.2M", "~200 FPS", "엣지"),
            ("YOLOv8m", "Medium", "25.9M", "~150 FPS", "권장"),
            ("YOLOv8l", "Large", "43.7M", "~100 FPS", "고성능"),
            ("YOLOv8x", "X-Large", "68.2M", "~80 FPS", "최고정확도")
        ]

        for col, (name, size, params, fps, use) in zip(model_cols, models):
            with col:
                st.markdown(f"""
                <div style="background:#f0f8ff; padding:10px; border-radius:5px; text-align:center;">
                    <h5>{name}</h5>
                    <p style="font-size:11px; margin:2px;">파라미터: {params}</p>
                    <p style="font-size:11px; margin:2px;">{fps}</p>
                    <p style="font-size:10px; color:#666;">{use}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        with st.expander("📦 YOLOv8 설치 및 사용", expanded=True):
            st.code("""
# 1. 설치
pip install ultralytics

# 2. 기본 사용
from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO('yolov8m.pt')  # 또는 yolov8n.pt, yolov8s.pt 등

# 이미지 추론
results = model('road_video.mp4')  # 비디오/이미지/폴더 경로

# 결과 접근
for result in results:
    boxes = result.boxes  # 바운딩 박스
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # 좌표
        conf = box.conf[0]             # 신뢰도
        cls = box.cls[0]               # 클래스
        label = model.names[int(cls)]   # 클래스 이름

        print(f"{label}: {conf:.2f}")

# 3. 실시간 웹캠
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()  # 바운딩 박스 그리기
    cv2.imshow('YOLOv8', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            """, language='python')

        with st.expander("🎨 커스텀 시각화", expanded=True):
            st.code("""
def draw_custom_boxes(frame, results, conf_threshold=0.5):
    \"\"\"커스텀 바운딩 박스 그리기\"\"\"

    # 클래스별 색상 정의
    colors = {
        'car': (0, 255, 0),        # 초록
        'truck': (255, 255, 0),    # 노랑
        'person': (0, 0, 255),     # 빨강
        'bicycle': (255, 0, 255),  # 자홍
        'motorcycle': (255, 128, 0) # 주황
    }

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 신뢰도 필터링
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            # 좌표 및 클래스
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = result.names[cls]

            # 색상 선택
            color = colors.get(label, (255, 255, 255))

            # 바운딩 박스 (두께 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 라벨 배경
            label_text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

            # 라벨 텍스트
            cv2.putText(frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return frame


# 사용 예시
cap = cv2.VideoCapture('road_video.mp4')
model = YOLO('yolov8m.pt')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 추론
    results = model(frame, conf=0.3)  # 신뢰도 0.3 이상만

    # 커스텀 시각화
    frame = draw_custom_boxes(frame, results, conf_threshold=0.5)

    cv2.imshow('Custom Visualization', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            """, language='python')

        with st.expander("⚡ 성능 최적화", expanded=True):
            st.markdown("""
            ### 1. 입력 크기 조정

            ```python
            # 작은 입력 → 빠르지만 정확도 낮음
            results = model(frame, imgsz=320)  # 기본 640

            # 큰 입력 → 느리지만 정확도 높음
            results = model(frame, imgsz=1280)
            ```

            ### 2. 배치 처리

            ```python
            # 여러 프레임 동시 처리
            frames = [frame1, frame2, frame3]
            results = model(frames, batch=3)
            ```

            ### 3. TensorRT 최적화

            ```python
            # ONNX 변환
            model.export(format='onnx')

            # TensorRT 엔진 빌드 (NVIDIA GPU)
            model.export(format='engine')  # .engine 파일 생성

            # TensorRT 모델 로드 (10배 빠름!)
            model = YOLO('yolov8m.engine')
            ```

            ### 4. 하프 정밀도 (FP16)

            ```python
            # GPU 메모리 절약 + 속도 향상
            results = model(frame, half=True)
            ```

            ### 5. 프레임 스킵

            ```python
            frame_skip = 2  # 2프레임마다 처리
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                frame_count += 1

                if frame_count % frame_skip == 0:
                    results = model(frame)
            ```
            """)

    def _render_bytetrack(self):
        """ByteTrack 객체 추적"""
        st.subheader("🔗 ByteTrack 객체 추적")

        st.markdown("""
        **객체 추적 (Object Tracking)**: 프레임 간 같은 객체에 일관된 ID 부여

        **필요성**:
        - 차량 행동 예측 (속도, 방향)
        - 차간 거리 모니터링
        - 위험 차량 식별

        **ByteTrack 특징**:
        - SOTA 성능 (MOT20 기준 80.3% MOTA)
        - 낮은 신뢰도 객체도 활용
        - 빠른 속도 (~30 FPS)
        """)

        st.markdown("---")

        # ByteTrack 알고리즘
        with st.expander("🧠 ByteTrack 알고리즘", expanded=True):
            st.markdown("""
            ### 기존 Tracking의 문제점

            **전통적 방식**: 높은 신뢰도 탐지만 사용
            ```
            YOLOv8 탐지 → conf > 0.7 필터링 → Tracking
            ```

            **문제**:
            - 가려진 객체 (낮은 신뢰도) 무시
            - ID 스위칭 빈번 발생
            - 프레임 간 불연속

            ---

            ### ByteTrack의 해결책

            **2-Stage 매칭**:

            ```
            YOLOv8 탐지
                ↓
            ┌─────────────┬─────────────┐
            │ 높은 신뢰도  │ 낮은 신뢰도  │
            │ (conf > 0.7)│ (0.1 < conf│
            │             │     < 0.7)  │
            └─────────────┴─────────────┘
                ↓              ↓
            [1차 매칭]   [2차 매칭]
            기존 트랙과   1차에서 안 맞은
            매칭          트랙과 매칭
                ↓
            최종 트랙 업데이트
            ```

            ---

            ### IoU (Intersection over Union) 매칭

            ```python
            def iou(box1, box2):
                \"\"\"두 박스의 IoU 계산\"\"\"
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection

                return intersection / union if union > 0 else 0


            # 매칭 예시
            detection = [100, 100, 200, 200]  # 새 탐지
            track = [105, 105, 205, 205]      # 기존 트랙

            iou_score = iou(detection, track)  # 0.81 (높음 → 같은 객체!)

            if iou_score > 0.5:
                track.update(detection)  # 트랙 업데이트
            ```

            ---

            ### 칼만 필터 예측

            ```python
            class KalmanTracker:
                def __init__(self, bbox):
                    self.kf = KalmanFilter(dim_x=7, dim_z=4)
                    # 상태: [x, y, s, r, vx, vy, vs]
                    # 관측: [x, y, s, r]
                    self.kf.x[:4] = bbox

                def predict(self):
                    \"\"\"다음 프레임 위치 예측\"\"\"
                    self.kf.predict()
                    return self.kf.x[:4]

                def update(self, bbox):
                    \"\"\"관측값으로 업데이트\"\"\"
                    self.kf.update(bbox)


            # 사용 예시
            tracker = KalmanTracker([100, 100, 50, 50])

            # 프레임 t
            predicted = tracker.predict()  # [102, 105, 50, 50]

            # 프레임 t+1 (실제 탐지)
            detected = [103, 106, 51, 51]
            tracker.update(detected)
            ```
            """)

        with st.expander("📦 ByteTrack 설치 및 사용", expanded=True):
            st.code("""
# 1. 설치
pip install bytetrack

# 2. 기본 사용
from bytetrack import BYTETracker
from ultralytics import YOLO
import numpy as np

# YOLOv8 + ByteTrack 초기화
model = YOLO('yolov8m.pt')
tracker = BYTETracker(
    track_thresh=0.5,      # 트랙 신뢰도 임계값
    track_buffer=30,       # 최대 프레임 버퍼
    match_thresh=0.8,      # 매칭 IoU 임계값
    frame_rate=30          # 비디오 FPS
)

cap = cv2.VideoCapture('road_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 탐지
    results = model(frame)

    # ByteTrack 형식으로 변환
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # [x1, y1, x2, y2, score, class]
            detections.append([x1, y1, x2, y2, conf, cls])

    detections = np.array(detections)

    # ByteTrack 업데이트
    if len(detections) > 0:
        online_targets = tracker.update(detections, [frame.shape[0], frame.shape[1]])

        # 트랙 그리기
        for track in online_targets:
            tlwh = track.tlwh  # [top, left, width, height]
            track_id = track.track_id

            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])

            # 바운딩 박스 + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('ByteTrack', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            """, language='python')

        st.markdown("---")

        # 트랙 정보 활용
        with st.expander("📊 트랙 정보 활용", expanded=True):
            st.markdown("""
            ### 1. 속도 계산

            ```python
            class VelocityTracker:
                def __init__(self):
                    self.tracks = {}  # {track_id: [positions]}

                def update(self, track_id, bbox, fps=30):
                    if track_id not in self.tracks:
                        self.tracks[track_id] = []

                    # 중심점 저장
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    self.tracks[track_id].append((cx, cy))

                    # 최근 10 프레임만 유지
                    if len(self.tracks[track_id]) > 10:
                        self.tracks[track_id].pop(0)

                    # 속도 계산 (픽셀/초)
                    if len(self.tracks[track_id]) >= 2:
                        p1 = self.tracks[track_id][-2]
                        p2 = self.tracks[track_id][-1]

                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        distance = np.sqrt(dx**2 + dy**2)

                        velocity = distance * fps  # 픽셀/초

                        # 픽셀 → 미터 변환 (카메라 캘리브레이션 필요)
                        velocity_mps = velocity * 0.05  # 예: 0.05m/픽셀
                        velocity_kmh = velocity_mps * 3.6

                        return velocity_kmh

                    return 0.0


            # 사용 예시
            vel_tracker = VelocityTracker()

            for track in online_targets:
                velocity = vel_tracker.update(track.track_id, track.tlbr)
                print(f"ID {track.track_id}: {velocity:.1f} km/h")
            ```

            ---

            ### 2. 궤적 예측

            ```python
            def predict_trajectory(tracks, num_future_frames=5):
                \"\"\"선형 보간으로 미래 위치 예측\"\"\"
                if len(tracks) < 3:
                    return None

                # 최근 3개 위치
                recent = tracks[-3:]
                x_coords = [p[0] for p in recent]
                y_coords = [p[1] for p in recent]

                # 선형 회귀
                t = np.arange(len(recent))
                vx = np.polyfit(t, x_coords, 1)[0]
                vy = np.polyfit(t, y_coords, 1)[0]

                # 미래 위치 예측
                future_positions = []
                for i in range(1, num_future_frames + 1):
                    future_x = x_coords[-1] + vx * i
                    future_y = y_coords[-1] + vy * i
                    future_positions.append((future_x, future_y))

                return future_positions


            # 시각화
            for track_id, positions in vel_tracker.tracks.items():
                future = predict_trajectory(positions)
                if future:
                    for i, (fx, fy) in enumerate(future):
                        cv2.circle(frame, (int(fx), int(fy)),
                                   3, (255, 0, 0), -1)
            ```

            ---

            ### 3. 충돌 위험 감지

            ```python
            def check_collision_risk(ego_track, other_track, threshold=50):
                \"\"\"두 차량의 충돌 위험 계산\"\"\"

                # 현재 거리
                ego_center = ((ego_track[0] + ego_track[2]) / 2,
                              (ego_track[1] + ego_track[3]) / 2)
                other_center = ((other_track[0] + other_track[2]) / 2,
                                (other_track[1] + other_track[3]) / 2)

                distance = np.sqrt(
                    (ego_center[0] - other_center[0])**2 +
                    (ego_center[1] - other_center[1])**2
                )

                # TTC (Time To Collision)
                ego_vel = vel_tracker.update(ego_id, ego_track)
                other_vel = vel_tracker.update(other_id, other_track)

                relative_vel = abs(ego_vel - other_vel)

                if relative_vel > 0:
                    ttc = distance / relative_vel
                else:
                    ttc = float('inf')

                # 위험 판정
                if ttc < 2.0:  # 2초 이내
                    return "HIGH", ttc
                elif ttc < 5.0:
                    return "MEDIUM", ttc
                else:
                    return "LOW", ttc
            ```
            """)

    def _render_ipm(self):
        """IPM 거리 추정"""
        st.subheader("📏 IPM (Inverse Perspective Mapping) 거리 추정")

        st.markdown("""
        **IPM**: 카메라 영상을 Bird's Eye View(BEV)로 변환하여 실제 거리 계산

        **목적**:
        - 픽셀 좌표 → 실제 미터 단위 변환
        - 차간 거리 정확히 계산
        - 충돌 위험 정량화
        """)

        st.markdown("---")

        with st.expander("🎯 IPM 원리", expanded=True):
            st.markdown("""
            ### 카메라 모델

            **핀홀 카메라 모델**:

            ```
            3D 월드 좌표 (X, Y, Z)
                ↓
            [카메라 외부 파라미터]
            회전(R), 이동(t)
                ↓
            카메라 좌표계 (Xc, Yc, Zc)
                ↓
            [카메라 내부 파라미터]
            초점거리(f), 주점(cx, cy)
                ↓
            2D 이미지 좌표 (u, v)
            ```

            **투영 방정식**:
            ```
            u = fx * (Xc / Zc) + cx
            v = fy * (Yc / Zc) + cy
            ```

            ---

            ### 역변환 (IPM)

            **가정**: Z=0 (평면 도로)

            ```python
            def pixel_to_world(u, v, camera_matrix, extrinsics):
                \"\"\"픽셀 좌표 → 월드 좌표 (Z=0)\"\"\"

                # 내부 파라미터
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

                # 정규화 좌표
                xn = (u - cx) / fx
                yn = (v - cy) / fy

                # 외부 파라미터 (회전 + 이동)
                R = extrinsics[:3, :3]
                t = extrinsics[:3, 3]

                # 역변환
                # [X, Y, Z]^T = R^-1 * (s * [xn, yn, 1]^T - t)
                # Z=0 조건 사용

                # ... (복잡한 행렬 계산) ...

                return X, Y  # 미터 단위
            ```

            ---

            ### 간단한 근사 방법

            ```python
            def simple_distance_estimation(bbox_bottom_y, camera_height=1.5):
                \"\"\"박스 하단 y좌표로 거리 근사\"\"\"

                # 카메라 높이: 1.5m (지면에서)
                # 이미지 높이: 720 픽셀
                # 수평선 (vanishing point): y=360

                horizon_y = 360
                image_height = 720

                # y좌표가 수평선에 가까울수록 멀리 있음
                if bbox_bottom_y <= horizon_y:
                    return float('inf')  # 하늘/배경

                # 간단한 비례식
                # distance ∝ 1 / (bbox_bottom_y - horizon_y)

                distance = (image_height - horizon_y) / (bbox_bottom_y - horizon_y)
                distance *= 50  # 스케일 조정 (캘리브레이션 필요)

                return distance  # 미터


            # 사용 예시
            for track in online_targets:
                tlwh = track.tlwh
                bottom_y = tlwh[1] + tlwh[3]

                distance = simple_distance_estimation(bottom_y)

                print(f"ID {track.track_id}: {distance:.1f}m")
            ```
            """)

        with st.expander("🎥 카메라 캘리브레이션", expanded=True):
            st.markdown("""
            ### OpenCV 체스보드 캘리브레이션

            ```python
            import cv2
            import numpy as np
            import glob

            # 1. 체스보드 이미지 촬영 (20-30장)
            # 체스보드 크기: 9x6 (내부 코너 수)

            # 2. 코너 점 찾기
            objp = np.zeros((6*9, 3), np.float32)
            objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

            objpoints = []  # 3D 점
            imgpoints = []  # 2D 점

            images = glob.glob('calibration/*.jpg')

            for fname in images:
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 코너 찾기
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)

            # 3. 캘리브레이션
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            print("Camera Matrix:")
            print(camera_matrix)
            # [[fx,  0, cx],
            #  [ 0, fy, cy],
            #  [ 0,  0,  1]]

            print("\\nDistortion Coefficients:")
            print(dist_coeffs)
            # [k1, k2, p1, p2, k3]

            # 4. 왜곡 보정
            img = cv2.imread('test.jpg')
            h, w = img.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix, dist_coeffs, (w, h), 1, (w, h)
            )

            undistorted = cv2.undistort(img, camera_matrix, dist_coeffs,
                                        None, new_camera_matrix)

            cv2.imshow('Undistorted', undistorted)
            ```

            ---

            ### 수동 캘리브레이션 (간단)

            ```python
            # 실제 측정으로 파라미터 설정

            # 1. 카메라 높이 측정
            camera_height = 1.5  # 미터

            # 2. 알려진 거리에 물체 배치 (예: 10m, 20m)
            # 3. 픽셀 y좌표 기록
            # 4. 룩업 테이블 생성

            distance_lut = {
                680: 5,    # y=680 픽셀 → 5미터
                650: 10,   # y=650 픽셀 → 10미터
                620: 15,
                590: 20,
                560: 25,
                530: 30
            }

            def lookup_distance(y):
                \"\"\"LUT로 거리 추정\"\"\"
                # 선형 보간
                keys = sorted(distance_lut.keys(), reverse=True)
                for i in range(len(keys)-1):
                    if keys[i] >= y >= keys[i+1]:
                        y1, d1 = keys[i], distance_lut[keys[i]]
                        y2, d2 = keys[i+1], distance_lut[keys[i+1]]
                        # 선형 보간
                        distance = d1 + (y - y1) * (d2 - d1) / (y2 - y1)
                        return distance
                return distance_lut[keys[-1]]
            ```
            """)

        st.markdown("---")

        # 전체 예시
        with st.expander("📋 거리 추정 전체 코드", expanded=False):
            st.code("""
import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import BYTETracker

# 모델 초기화
model = YOLO('yolov8m.pt')
tracker = BYTETracker()

# 간단한 거리 추정 함수
def estimate_distance(bbox_bottom_y, image_height=720):
    horizon_y = image_height * 0.5  # 수평선: 중간

    if bbox_bottom_y <= horizon_y:
        return float('inf')

    distance = (image_height - horizon_y) / (bbox_bottom_y - horizon_y)
    distance *= 50  # 스케일 (캘리브레이션 필요)

    return min(distance, 100)  # 최대 100m


# 비디오 처리
cap = cv2.VideoCapture('road_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # 1. YOLOv8 탐지
    results = model(frame)

    # 2. ByteTrack 변환
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls])

    detections = np.array(detections)

    # 3. ByteTrack 추적
    if len(detections) > 0:
        online_targets = tracker.update(detections, [height, width])

        for track in online_targets:
            tlwh = track.tlwh
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
            track_id = track.track_id

            # 4. 거리 추정
            bottom_y = y2
            distance = estimate_distance(bottom_y, height)

            # 5. 시각화
            color = (0, 255, 0) if distance > 20 else (0, 165, 255) if distance > 10 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id} {distance:.1f}m"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Distance Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            """, language='python')

    def _render_object_full_code(self):
        """객체 탐지/추적 전체 코드"""
        st.subheader("📋 전체 통합 코드")

        st.markdown("""
        **통합 기능**:
        - YOLOv8 탐지
        - ByteTrack 추적
        - 거리 추정
        - 속도 계산
        - 충돌 위험 분석
        """)

        st.code("""
# Week 10: 객체 탐지 및 추적 (완전판)

import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import BYTETracker
from collections import defaultdict


class AutonomousDrivingPerception:
    \"\"\"자율주행 인식 시스템\"\"\"

    def __init__(self, model_path='yolov8m.pt'):
        # YOLOv8 모델
        self.model = YOLO(model_path)

        # ByteTrack
        self.tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30
        )

        # 트랙 히스토리
        self.track_history = defaultdict(list)

        # 프레임 카운터
        self.frame_count = 0

    def estimate_distance(self, bbox_bottom_y, image_height):
        \"\"\"간단한 거리 추정\"\"\"
        horizon_y = image_height * 0.5

        if bbox_bottom_y <= horizon_y:
            return float('inf')

        distance = (image_height - horizon_y) / (bbox_bottom_y - horizon_y)
        distance *= 50

        return min(distance, 100)

    def calculate_velocity(self, track_id, current_pos, fps=30):
        \"\"\"속도 계산 (km/h)\"\"\"
        self.track_history[track_id].append(current_pos)

        # 최근 10프레임만 유지
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id].pop(0)

        if len(self.track_history[track_id]) >= 2:
            p1 = self.track_history[track_id][-2]
            p2 = self.track_history[track_id][-1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance_px = np.sqrt(dx**2 + dy**2)

            velocity_pxps = distance_px * fps
            velocity_mps = velocity_pxps * 0.05  # 픽셀→미터 (캘리브레이션)
            velocity_kmh = velocity_mps * 3.6

            return velocity_kmh

        return 0.0

    def assess_risk(self, distance, velocity):
        \"\"\"위험도 평가\"\"\"
        # TTC (Time To Collision)
        if velocity > 0:
            ttc = distance / (velocity / 3.6)  # km/h → m/s
        else:
            ttc = float('inf')

        if ttc < 2.0:
            return "HIGH", ttc
        elif ttc < 5.0:
            return "MEDIUM", ttc
        else:
            return "LOW", ttc

    def process_frame(self, frame):
        \"\"\"프레임 처리\"\"\"
        self.frame_count += 1
        height, width = frame.shape[:2]

        # 1. YOLOv8 탐지
        results = self.model(frame, verbose=False)

        # 2. ByteTrack 변환
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append([x1, y1, x2, y2, conf, cls])

        detections = np.array(detections)

        # 3. ByteTrack 추적
        if len(detections) > 0:
            online_targets = self.tracker.update(detections, [height, width])

            for track in online_targets:
                tlwh = track.tlwh
                x1, y1 = int(tlwh[0]), int(tlwh[1])
                x2, y2 = int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])
                track_id = track.track_id

                # 중심점
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 4. 거리 추정
                distance = self.estimate_distance(y2, height)

                # 5. 속도 계산
                velocity = self.calculate_velocity(track_id, (cx, cy))

                # 6. 위험도 평가
                risk_level, ttc = self.assess_risk(distance, velocity)

                # 7. 시각화
                # 위험도별 색상
                colors = {
                    "HIGH": (0, 0, 255),      # 빨강
                    "MEDIUM": (0, 165, 255),  # 주황
                    "LOW": (0, 255, 0)        # 초록
                }
                color = colors.get(risk_level, (255, 255, 255))

                # 바운딩 박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 정보 텍스트
                label = f"ID:{track_id} {distance:.1f}m {velocity:.0f}km/h"
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 위험도 표시
                if risk_level != "LOW":
                    risk_text = f"{risk_level} (TTC:{ttc:.1f}s)"
                    cv2.putText(frame, risk_text, (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 궤적 그리기
                if len(self.track_history[track_id]) > 1:
                    points = np.array(self.track_history[track_id], dtype=np.int32)
                    cv2.polylines(frame, [points], False, color, 2)

        # FPS 표시
        fps_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame


def main():
    # 시스템 초기화
    perception = AutonomousDrivingPerception('yolov8m.pt')

    # 비디오 로드
    cap = cv2.VideoCapture('road_video.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        result = perception.process_frame(frame)

        # 결과 표시
        cv2.imshow('Autonomous Driving Perception', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
        """, language='python')

        st.markdown("---")

        st.success("""
        ✅ **완성된 기능**:
        - YOLOv8 실시간 객체 탐지
        - ByteTrack ID 유지 추적
        - 거리 추정 (IPM 근사)
        - 속도 계산 (km/h)
        - TTC 기반 충돌 위험 분석
        - 색상별 위험도 표시
        - 궤적 시각화

        다음 탭에서 이 모듈들을 통합한 전체 파이프라인을 구현합니다!
        """)

    # ==================== Tab 4-7은 다음 메시지에서 계속 ====================

    def render_integrated_pipeline(self):
        """통합 파이프라인 (간단 버전)"""
        st.header("🔗 통합 파이프라인")
        st.info("차선 인식 + 객체 탐지/추적을 통합한 완전한 시스템입니다. 전체 코드는 Colab 노트북을 참조하세요.")

    def render_3d_visualization(self):
        """3D 시각화 (간단 버전)"""
        st.header("📐 3D 시각화 (BEV)")
        st.info("Bird's Eye View 변환 및 3D 바운딩 박스 시각화입니다. 전체 구현은 Colab 노트북을 참조하세요.")

    def render_simulator(self):
        """고급 시뮬레이터 (간단 버전)"""
        st.header("🎮 고급 시뮬레이터")
        st.info("교차로, 신호등, 날씨 효과를 포함한 시뮬레이터입니다. 전체 구현은 Colab 노트북을 참조하세요.")

    def render_deployment(self):
        """실전 배포 (간단 버전)"""
        st.header("💻 실전 배포")
        st.info("TensorRT 최적화 및 Edge 디바이스 배포 가이드입니다. 전체 구현은 Colab 노트북을 참조하세요.")
