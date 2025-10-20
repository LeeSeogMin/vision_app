"""
Lab 04: ControlNet 실습
조건부 생성을 위한 ControlNet 활용
"""

import streamlit as st


def render():
    st.subheader("Lab 04: ControlNet 실습")
    st.markdown("""
    ### 학습 목표
    - ControlNet의 원리 이해하기
    - 다양한 조건 신호 활용하기
    - 구도를 제어한 이미지 생성

    ### ControlNet 타입
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Canny Edge**
        - 윤곽선 기반
        - 명확한 형태 보존
        - 스케치 → 이미지
        """)

    with col2:
        st.markdown("""
        **OpenPose**
        - 포즈 기반
        - 인체 자세 제어
        - 캐릭터 생성
        """)

    with col3:
        st.markdown("""
        **Depth**
        - 깊이 정보 기반
        - 3D 구조 보존
        - 공간감 제어
        """)

    st.markdown("### 실습 시나리오")
    st.info("""
    1. 입력 이미지에서 조건 신호 추출 (예: Canny edge)
    2. ControlNet 파이프라인 로드
    3. 프롬프트와 조건 신호로 이미지 생성
    4. 결과 비교 및 분석
    """)

    with st.expander("📝 예제 코드"):
        st.code("""
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

# ControlNet 모델 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Canny edge 추출
image = Image.open("input.jpg")
image_np = np.array(image)
edges = cv2.Canny(image_np, 100, 200)
edges = Image.fromarray(edges)

# ControlNet 생성
output = pipe(
    prompt="A beautiful landscape painting",
    image=edges,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

output.save("controlnet_output.png")
        """, language="python")
