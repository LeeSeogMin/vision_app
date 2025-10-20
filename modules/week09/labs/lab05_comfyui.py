"""
Lab 05: ComfyUI 워크플로우
노드 기반 파이프라인 작성 및 활용
"""

import streamlit as st


def render():
    st.subheader("Lab 05: ComfyUI 워크플로우")
    st.markdown("""
    ### 학습 목표
    - ComfyUI의 노드 시스템 이해하기
    - 기본 워크플로우 구성하기
    - 재현 가능한 파이프라인 만들기

    ### ComfyUI 장점
    - 🔧 **시각적 편집**: 노드 기반 직관적 인터페이스
    - 🔄 **재현성**: 워크플로우 저장 및 공유
    - 🎯 **유연성**: 복잡한 파이프라인 구성 가능
    - 📊 **협업**: 팀 간 워크플로우 공유
    """)

    st.markdown("### 기본 노드 구성")

    st.code("""
    [Load Checkpoint]
         ↓
    [CLIP Text Encode (Prompt)]
         ↓
    [KSampler]
         ↓
    [VAE Decode]
         ↓
    [Save Image]
    """, language="text")

    st.markdown("### 실습 워크플로우")

    with st.expander("1️⃣ 기본 Text-to-Image"):
        st.markdown("""
        **필요 노드**:
        - Load Checkpoint: 모델 로드
        - CLIP Text Encode: 프롬프트 인코딩
        - KSampler: 샘플링
        - VAE Decode: 잠재 공간 → 이미지
        - Save Image: 결과 저장
        """)

    with st.expander("2️⃣ ControlNet 워크플로우"):
        st.markdown("""
        **추가 노드**:
        - Load Image: 입력 이미지
        - ControlNet Preprocessor: 조건 신호 추출
        - Apply ControlNet: ControlNet 적용
        """)

    with st.expander("3️⃣ LoRA 적용"):
        st.markdown("""
        **추가 노드**:
        - Load LoRA: LoRA 가중치 로드
        - LoRA Stack: 여러 LoRA 조합
        """)

    st.markdown("### 실습 과제")
    st.info("""
    **목표**: 3가지 스타일의 프로필 이미지 생성

    1. 동일 프롬프트, 시드, 스케줄러 고정
    2. 스타일 LoRA만 변경
    3. 결과 비교 및 스크린샷 저장

    **제출물**: 워크플로우 JSON + 생성 이미지 3장 + 비교 분석
    """)

    with st.expander("📝 ComfyUI 설치 가이드"):
        st.code("""
# ComfyUI 설치
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

# 실행
python main.py

# 브라우저에서 접속
# http://127.0.0.1:8188
        """, language="bash")
