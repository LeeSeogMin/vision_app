"""
Lab 01: 기본 Diffusion 모델 실습
Stable Diffusion을 사용한 기본 텍스트-이미지 생성
"""

import streamlit as st


def render():
    st.subheader("Lab 01: 기본 Diffusion 모델")
    st.markdown("""
    ### 학습 목표
    - Stable Diffusion 파이프라인 이해하기
    - 기본 텍스트-이미지 생성 실습
    - 파라미터(steps, CFG scale) 영향 파악

    ### 실습 내용
    1. Stable Diffusion 파이프라인 로드
    2. 간단한 프롬프트로 이미지 생성
    3. 파라미터 조정 및 결과 비교
    """)

    st.info("💡 **팁**: Google Colab에서 실행하면 GPU를 활용하여 빠른 생성이 가능합니다.")

    with st.expander("📝 예제 코드"):
        st.code("""
from diffusers import StableDiffusionPipeline
import torch

# 모델 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 이미지 생성
prompt = "A beautiful sunset over mountains, digital art"
image = pipe(
    prompt,
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]

image.save("output.png")
        """, language="python")
