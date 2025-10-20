"""
Lab 03: 스케줄러 비교
다양한 스케줄러의 특성과 성능 비교
"""

import streamlit as st


def render():
    st.subheader("Lab 03: 스케줄러 비교")
    st.markdown("""
    ### 학습 목표
    - 다양한 스케줄러의 특성 이해하기
    - 스케줄러별 품질과 속도 비교
    - 용도에 맞는 스케줄러 선택하기

    ### 주요 스케줄러
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **DDIM**
        - 빠른 샘플링
        - 적은 스텝에서 우수
        - 안정적인 결과
        """)

        st.markdown("""
        **DPMSolver++**
        - 품질/속도 균형 우수
        - 최신 기본값
        - 범용적으로 사용
        """)

    with col2:
        st.markdown("""
        **Euler**
        - 샤프한 디테일
        - 스타일 특화
        - 빠른 수렴
        """)

        st.markdown("""
        **Euler Ancestral**
        - 창의적인 결과
        - 변화가 큼
        - 실험적 용도
        """)

    with st.expander("📝 예제 코드"):
        st.code("""
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)

# 파이프라인 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 스케줄러 변경
schedulers = {
    "DDIM": DDIMScheduler,
    "DPMSolver": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "EulerA": EulerAncestralDiscreteScheduler
}

for name, scheduler_class in schedulers.items():
    pipe.scheduler = scheduler_class.from_config(
        pipe.scheduler.config
    )

    image = pipe(
        prompt="A serene landscape",
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(42)
    ).images[0]

    image.save(f"output_{name}.png")
        """, language="python")
