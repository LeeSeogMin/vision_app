"""
Lab 02: 프롬프트 엔지니어링
효과적인 프롬프트 작성 및 네거티브 프롬프트 활용
"""

import streamlit as st


def render():
    st.subheader("Lab 02: 프롬프트 엔지니어링")
    st.markdown("""
    ### 학습 목표
    - 효과적인 프롬프트 구조 이해하기
    - 네거티브 프롬프트의 역할 파악
    - 스타일 토큰 활용법 익히기

    ### 프롬프트 구성 요소
    1. **주체(Subject)**: 무엇을 그릴 것인가
    2. **스타일(Style)**: 어떤 스타일로 그릴 것인가
    3. **조명(Lighting)**: 조명 설정
    4. **구도(Composition)**: 구도 및 앵글
    5. **품질 토큰**: ultra-detailed, 8k, high quality
    """)

    st.markdown("### 프롬프트 템플릿")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**기본 템플릿**")
        st.code("""
{subject}, {style}, {lighting},
{composition}, ultra-detailed, 8k
        """, language="text")

    with col2:
        st.markdown("**네거티브 프롬프트**")
        st.code("""
low quality, blurry, artifacts,
deformed, extra fingers, bad anatomy
        """, language="text")

    with st.expander("📝 예제 코드"):
        st.code("""
prompt = '''
Portrait of a wise old wizard, fantasy art style,
cinematic lighting, close-up shot,
ultra-detailed, 8k, sharp focus
'''

negative_prompt = '''
low quality, blurry, artifacts, deformed hands,
extra fingers, bad anatomy, poorly drawn face
'''

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]
        """, language="python")
