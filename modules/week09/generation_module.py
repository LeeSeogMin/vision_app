"""
Week 9: 생성 모델 이론 + Stable Diffusion (Streamlit Module)

기능 탭
- 📚 개념 소개
- 🧪 Diffusers 데모
- ✍️ 프롬프트 엔지니어링
- ⏱️ 스케줄러 비교
- 🧩 ControlNet/Adapter
- 🗺️ ComfyUI 가이드
"""

from typing import Dict, Any, Optional
import os
import io

import streamlit as st
from PIL import Image

from core.base_processor import BaseImageProcessor


class GenerationModule(BaseImageProcessor):
    """Week 9 Streamlit UI module for text-to-image generation."""

    def __init__(self):
        super().__init__()
        self.name = 'Week 9: Text-to-Image Generation'

    def render(self):
        st.title('🧪 Week 9: Stable Diffusion & Diffusers')
        st.caption('무료 환경(Colab) + Hugging Face Diffusers 중심의 실습')

        # 환경 체크 패널
        self._display_environment_status()

        tabs = st.tabs([
            '📚 개념 소개',
            '🛠️ 방법(Colab/로컬)',
            '🧪 Diffusers 데모',
            '✍️ 프롬프트 엔지니어링',
            '⏱️ 스케줄러 비교',
            '🧩 ControlNet/Adapter',
            '🗺️ ComfyUI 가이드',
        ])

        with tabs[0]:
            self.render_theory()
        with tabs[1]:
            self.render_method()
        with tabs[2]:
            self.render_diffusers_demo()
        with tabs[3]:
            self.render_prompt_lab()
        with tabs[4]:
            self.render_scheduler_lab()
        with tabs[5]:
            self.render_controlnet_lab()
        with tabs[6]:
            self.render_comfyui_guide()

    def render_theory(self):
        st.header('📚 생성 모델 이론 + Stable Diffusion')

        st.markdown("""
        ### 🎯 학습 목표
        - Diffusion 모델의 원리 이해
        - Stable Diffusion 파이프라인 구성 요소 파악
        - VAE와 잠재 공간의 역할 이해
        - 프롬프트 엔지니어링 기법 습득
        """)

        st.markdown('---')

        # 1. 생성 모델 개론
        st.subheader('🌟 1. 생성 모델의 발전')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **생성 모델 타임라인**
            ```
            GAN (2014)
              ↓
            VAE (2013-2016)
              ↓
            Diffusion Models (2020)
              ↓
            Stable Diffusion (2022)
            ```
            """)

        with col2:
            st.markdown("""
            | 모델 | 장점 | 단점 |
            |------|------|------|
            | **GAN** | 빠른 생성 | 학습 불안정 |
            | **VAE** | 안정적 학습 | 흐릿한 결과 |
            | **Diffusion** | 고품질 | 느린 생성 |
            """)

        st.markdown('---')

        # 2. Diffusion 모델 원리
        st.subheader('🔬 2. Diffusion 모델의 원리')

        st.markdown("""
        **Forward Process (순방향 과정)**
        ```
        원본 이미지 → 노이즈 추가 → ... → 순수 노이즈
        x₀          x₁              ...   xₜ
        ```

        **Reverse Process (역방향 과정)**
        ```
        순수 노이즈 → 노이즈 제거 → ... → 원본 이미지
        xₜ           xₜ₋₁            ...   x₀
        ```

        **학습 목표**: 노이즈를 예측하는 신경망 학습
        """)

        with st.expander('📖 수식 설명'):
            st.latex(r'q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)')
            st.latex(r'L = \mathbb{E}[||\epsilon - \epsilon_\theta(x_t, t)||^2]')
            st.caption('노이즈 예측 손실 함수')

        st.markdown('---')

        # 3. Stable Diffusion 아키텍처
        st.subheader('🏗️ 3. Stable Diffusion 아키텍처')

        st.markdown("""
        **전체 파이프라인**
        ```
        텍스트 프롬프트
            ↓
        CLIP Text Encoder → Text Embeddings
                             ↓
        랜덤 노이즈 → U-Net (조건부 노이즈 예측) → 잠재 표현
                             ↑
                        Time Step
                             ↓
                        VAE Decoder → 최종 이미지
        ```
        """)

        # 3.1 VAE
        with st.expander('💡 VAE (Variational Autoencoder)'):
            st.markdown("""
            **목적**: 고차원 이미지 → 저차원 잠재 공간

            - **Encoder**: 512×512 이미지 → 64×64 잠재 표현 (8배 압축)
            - **Decoder**: 64×64 잠재 표현 → 512×512 이미지

            **장점**:
            - 메모리 효율성 (64배 감소)
            - 빠른 샘플링
            - 고품질 재구성
            """)

        # 3.2 CLIP Text Encoder
        with st.expander('💡 CLIP Text Encoder'):
            st.markdown("""
            **목적**: 텍스트 → 의미 벡터

            ```
            "A beautiful sunset" → [0.23, -0.45, 0.67, ...]
                                   (77 토큰 × 768 차원)
            ```

            - 텍스트와 이미지를 동일한 의미 공간으로 매핑
            - Cross-Attention을 통해 U-Net과 연결
            """)

        # 3.3 U-Net
        with st.expander('💡 U-Net'):
            st.markdown("""
            **구조**:
            ```
                Input (64×64)
                     ↓
               Encoder (DownBlock)
                32×32 → 16×16 → 8×8
                     ↓
               Middle Block
                     ↓
               Decoder (UpBlock)
                8×8 → 16×16 → 32×32
                     ↓
                Output (64×64)
            ```

            **Cross-Attention**:
            - Query: 이미지 특징 (U-Net)
            - Key/Value: 텍스트 임베딩 (CLIP)
            - → 텍스트-이미지 정렬
            """)

        st.markdown('---')

        # 4. 스케줄러
        st.subheader('⏱️ 4. 샘플링 스케줄러')

        scheduler_cols = st.columns(4)

        with scheduler_cols[0]:
            st.markdown("""
            **DDIM**
            - 빠른 샘플링
            - 20-50 스텝
            - 재현 가능
            """)

        with scheduler_cols[1]:
            st.markdown("""
            **DPM-Solver++**
            - 품질/속도 균형
            - 20-30 스텝
            - 최신 기본값
            """)

        with scheduler_cols[2]:
            st.markdown("""
            **Euler**
            - 샤프한 디테일
            - 20-40 스텝
            - 안정적
            """)

        with scheduler_cols[3]:
            st.markdown("""
            **Euler A**
            - 창의적 결과
            - 30-40 스텝
            - 실험적
            """)

        st.markdown('---')

        # 5. 프롬프트 엔지니어링
        st.subheader('✍️ 5. 프롬프트 엔지니어링')

        st.markdown("""
        **효과적인 프롬프트 구조**
        ```
        [주체] + [스타일] + [조명] + [구도] + [품질 토큰]
        ```

        **예시**:
        ```
        Portrait of a wise old wizard,        ← 주체
        fantasy art style,                    ← 스타일
        cinematic lighting,                   ← 조명
        close-up shot,                        ← 구도
        ultra-detailed, 8k, sharp focus       ← 품질 토큰
        ```
        """)

        prompt_cols = st.columns(2)

        with prompt_cols[0]:
            st.markdown("""
            **스타일 토큰**
            - `oil painting`
            - `digital art`
            - `photorealistic`
            - `cinematic lighting`
            - `8k resolution`
            """)

        with prompt_cols[1]:
            st.markdown("""
            **네거티브 프롬프트**
            - `low quality`
            - `blurry`
            - `artifacts`
            - `deformed hands`
            - `bad anatomy`
            """)

        st.markdown('---')

        # 6. 주요 파라미터
        st.subheader('🎚️ 6. 주요 파라미터')

        param_cols = st.columns(3)

        with param_cols[0]:
            st.markdown("""
            **CFG Scale**
            - 1-5: 창의적
            - 7-9: 균형 (권장)
            - 10-15: 충실, 과포화
            """)

        with param_cols[1]:
            st.markdown("""
            **Inference Steps**
            - 10-20: 빠름, 품질 저하
            - 25-35: 균형 (권장)
            - 40-50: 느림, 미미한 개선
            """)

        with param_cols[2]:
            st.markdown("""
            **Seed**
            - 동일 시드 = 재현 가능
            - 다른 시드 = 다른 결과
            - 실험: 시드 고정 후 비교
            """)

        st.markdown('---')

        # 7. ControlNet
        st.subheader('🧩 7. ControlNet')

        st.markdown("""
        **목적**: 추가 조건 신호로 생성 과정 제어

        ```
        텍스트 프롬프트 + 조건 이미지 → 구도가 제어된 결과
        ```
        """)

        controlnet_cols = st.columns(4)

        with controlnet_cols[0]:
            st.markdown("""
            **Canny Edge**
            - 윤곽선 기반
            - 명확한 형태
            - 스케치 → 이미지
            """)

        with controlnet_cols[1]:
            st.markdown("""
            **OpenPose**
            - 포즈 기반
            - 인체 자세 제어
            - 캐릭터 생성
            """)

        with controlnet_cols[2]:
            st.markdown("""
            **Depth**
            - 깊이 정보
            - 3D 구조 보존
            - 공간감 제어
            """)

        with controlnet_cols[3]:
            st.markdown("""
            **Scribble**
            - 낙서 기반
            - 빠른 스케치
            - 자유로운 표현
            """)

        st.markdown('---')

        # 8. LoRA
        st.subheader('🎨 8. 경량 미세튜닝 (LoRA)')

        st.markdown("""
        **LoRA (Low-Rank Adaptation)**

        - **전체 모델**: 4GB
        - **LoRA 가중치**: 10-100MB (400배 작음)

        **장점**:
        - 빠른 학습
        - 낮은 메모리 요구
        - 여러 LoRA 조합 가능
        - 스타일 미세 조정
        """)

        st.markdown('---')

        # 9. 안전 및 윤리
        st.subheader('🛡️ 9. 안전 및 윤리')

        safety_cols = st.columns(2)

        with safety_cols[0]:
            st.success("""
            **허용되는 사용**
            - ✅ 학습 목적
            - ✅ 개인 프로젝트
            - ✅ 연구 및 실험
            - ✅ 오픈소스 기여
            """)

        with safety_cols[1]:
            st.error("""
            **주의사항**
            - ⚠️ 상업적 사용 시 라이선스 확인
            - ❌ 저작권 침해 이미지 생성 금지
            - ❌ 유해 콘텐츠 생성 금지
            - ❌ 딥페이크 악용 방지
            """)

        st.markdown('---')

        # 10. 실습 환경
        st.subheader('💻 10. 실습 환경')

        env_cols = st.columns(2)

        with env_cols[0]:
            st.markdown("""
            **Google Colab (권장)**
            - GPU: T4 (무료)
            - 런타임 유형: GPU
            - 설치:
            ```bash
            pip install -q diffusers transformers accelerate torch --upgrade
            ```
            """)

        with env_cols[1]:
            st.markdown("""
            **로컬 환경**
            - GPU: 4GB VRAM 이상 권장
            - RAM: 8GB 이상
            - 저장 공간: 10GB 이상
            - Python 3.8+
            """)

        st.info("""
        💡 **팁**: GPU가 없으면 CPU로도 동작하지만 매우 느립니다.
        실습은 Google Colab의 무료 GPU 사용을 권장합니다.
        """)

    def _ensure_diffusers(self) -> bool:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
            return True
        except Exception:
            st.warning('⚠️ diffusers/torch 패키지가 필요합니다. Colab 또는 로컬에서 설치하세요: `pip install diffusers transformers accelerate torch --upgrade`')
            return False

    def _check_environment(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        try:
            import diffusers  # noqa: F401
            status['diffusers'] = True
        except Exception:
            status['diffusers'] = False
        try:
            import torch  # noqa: F401
            status['torch'] = True
        except Exception:
            status['torch'] = False
        try:
            import transformers  # noqa: F401
            status['transformers'] = True
        except Exception:
            status['transformers'] = False
        try:
            import accelerate  # noqa: F401
            status['accelerate'] = True
        except Exception:
            status['accelerate'] = False
        return status

    def _display_environment_status(self):
        status = self._check_environment()
        cols = st.columns(4)
        with cols[0]:
            st.success('✅ diffusers') if status.get('diffusers') else st.warning('⚠️ diffusers 미설치')
        with cols[1]:
            st.success('✅ torch') if status.get('torch') else st.warning('⚠️ torch 미설치')
        with cols[2]:
            st.success('✅ transformers') if status.get('transformers') else st.warning('⚠️ transformers 미설치')
        with cols[3]:
            st.success('✅ accelerate') if status.get('accelerate') else st.warning('⚠️ accelerate 미설치')

    def render_method(self):
        st.header('🛠️ 방법 (Colab/로컬)')
        st.markdown(
            """
            **Colab(권장)**
            1) 런타임: GPU(T4)
            2) 설치:
            ```bash
            pip install -q diffusers transformers accelerate torch --upgrade
            ```
            3) 모델: `runwayml/stable-diffusion-v1-5`

            **로컬(이 venv)**
            - `venv` 활성화 후 동일 설치
            - GPU가 없으면 CPU로 동작(느릴 수 있음)
            """
        )

    def render_diffusers_demo(self):
        st.header('🧪 Diffusers 데모 (CPU/GPU 환경 필요)')
        st.info('Colab에서 실행 권장. 로컬도 가능(venv)')

        if not self._ensure_diffusers():
            return

        model_id = st.selectbox('모델 선택', [
            'runwayml/stable-diffusion-v1-5',
            'stabilityai/sd-turbo'
        ], index=0, help='동일 파이프라인 사용 가능 모델 위주')

        scheduler_name = st.selectbox('스케줄러', [
            'DPMSolverMultistep',
            'DDIM',
            'Euler',
            'EulerAncestral'
        ], index=0)

        cols = st.columns(2)
        with cols[0]:
            steps = st.slider('Inference Steps', 5, 50, 25)
            guidance = st.slider('CFG Scale', 1.0, 12.0, 7.0)
            seed = st.number_input('Seed', min_value=0, max_value=2**31 - 1, value=42)
        with cols[1]:
            width = st.selectbox('Width', [512, 640, 768], index=0)
            height = st.selectbox('Height', [512, 640, 768], index=0)

        prompt = st.text_area('텍스트 프롬프트', 'A high quality portrait photo of a friendly teacher, studio lighting')
        negative = st.text_input('네거티브 프롬프트 (선택)', 'low quality, blurry, artifacts')

        if st.button('🚀 이미지 생성', type='primary', use_container_width=True):
            with st.spinner('Stable Diffusion으로 이미지를 생성 중...'):
                try:
                    import torch
                    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler

                    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

                    if scheduler_name == 'DPMSolverMultistep':
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                    elif scheduler_name == 'DDIM':
                        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    elif scheduler_name == 'Euler':
                        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
                    elif scheduler_name == 'EulerAncestral':
                        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    if torch.cuda.is_available():
                        pipe = pipe.to('cuda')

                    generator = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(int(seed))
                    image = pipe(
                        prompt,
                        negative_prompt=negative or None,
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance),
                        generator=generator,
                        height=int(height),
                        width=int(width),
                    ).images[0]

                    st.image(image, caption='Generated Image', use_container_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format='PNG')
                    st.download_button('📥 이미지 다운로드 (PNG)', buf.getvalue(), file_name='generated.png', mime='image/png', use_container_width=True)
                except Exception as e:
                    st.error(f'이미지 생성 실패: {e}')

    def render_prompt_lab(self):
        st.header('✍️ 프롬프트 엔지니어링 실습')

        st.markdown("""
        ### 🎯 학습 목표
        - 효과적인 프롬프트 구조 이해
        - 네거티브 프롬프트의 영향 파악
        - 스타일 토큰 활용법 익히기
        """)

        st.markdown('---')

        # 프롬프트 구성 요소
        st.subheader('📝 프롬프트 구성 요소')

        comp_cols = st.columns(5)

        with comp_cols[0]:
            st.markdown("""
            **1. 주체**
            - portrait
            - landscape
            - character
            - object
            """)

        with comp_cols[1]:
            st.markdown("""
            **2. 스타일**
            - photorealistic
            - oil painting
            - digital art
            - anime
            """)

        with comp_cols[2]:
            st.markdown("""
            **3. 조명**
            - cinematic
            - studio
            - natural
            - dramatic
            """)

        with comp_cols[3]:
            st.markdown("""
            **4. 구도**
            - close-up
            - wide shot
            - portrait
            - panoramic
            """)

        with comp_cols[4]:
            st.markdown("""
            **5. 품질**
            - 8k
            - detailed
            - sharp focus
            - high quality
            """)

        st.markdown('---')

        # 프롬프트 템플릿
        st.subheader('🎨 프롬프트 템플릿')

        template_tabs = st.tabs(['인물 사진', '풍경', '컨셉 아트', '일러스트'])

        with template_tabs[0]:
            st.code("""
Portrait of {subject},
{style} photography,
{lighting} lighting,
{composition} shot,
ultra-detailed, 8k, sharp focus,
professional photography

예시:
Portrait of a wise old wizard,
cinematic photography,
dramatic lighting with rim light,
close-up shot,
ultra-detailed, 8k, sharp focus,
professional photography
            """, language='text')

        with template_tabs[1]:
            st.code("""
{subject} landscape,
{time_of_day},
{weather} weather,
{style} style,
{composition},
ultra-detailed, 8k resolution

예시:
Mountain valley landscape,
golden hour sunset,
clear weather with volumetric lighting,
photorealistic style,
wide panoramic shot,
ultra-detailed, 8k resolution
            """, language='text')

        with template_tabs[2]:
            st.code("""
{subject} concept art,
{art_style},
{mood} atmosphere,
trending on artstation,
highly detailed,
digital painting

예시:
Futuristic city concept art,
sci-fi cyberpunk style,
neon-lit atmospheric mood,
trending on artstation,
highly detailed architectural design,
digital painting
            """, language='text')

        with template_tabs[3]:
            st.code("""
{subject} illustration,
{art_style} art style,
{color_palette},
{detail_level},
digital art

예시:
Cute cat character illustration,
anime art style,
vibrant pastel colors,
highly detailed with soft shading,
digital art
            """, language='text')

        st.markdown('---')

        # 네거티브 프롬프트
        st.subheader('🚫 네거티브 프롬프트')

        neg_cols = st.columns(3)

        with neg_cols[0]:
            st.markdown("""
            **일반적인 품질 문제**
            ```
            low quality, blurry, pixelated,
            artifacts, jpeg compression,
            noise, grainy, distorted
            ```
            """)

        with neg_cols[1]:
            st.markdown("""
            **인물/해부학적 문제**
            ```
            deformed hands, extra fingers,
            bad anatomy, poorly drawn face,
            mutation, malformed limbs,
            extra limbs, ugly
            ```
            """)

        with neg_cols[2]:
            st.markdown("""
            **구도/스타일 문제**
            ```
            cropped, out of frame,
            watermark, text, logo,
            duplicate, cloned face,
            cartoon (if not wanted)
            ```
            """)

        st.markdown('---')

        # Google Colab 코드
        st.subheader('💻 Google Colab 실습 코드')

        st.info('🔗 **권장**: 아래 코드를 Google Colab에 복사하여 무료 GPU로 실행하세요!')

        with st.expander('📋 전체 Colab 코드 보기'):
            st.code("""
# ========================================
# Week 9: 프롬프트 엔지니어링 실습
# Google Colab에서 실행하세요!
# ========================================

# 1. 패키지 설치
!pip install -q diffusers transformers accelerate torch

# 2. 라이브러리 임포트
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

# 3. 모델 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 4. 프롬프트 템플릿 정의
prompts = {
    "realistic": '''
        Portrait of a wise old wizard,
        cinematic photography,
        dramatic lighting with rim light,
        close-up shot,
        ultra-detailed, 8k, sharp focus,
        professional photography
    ''',

    "artistic": '''
        Portrait of a wise old wizard,
        oil painting style,
        soft warm lighting,
        classical composition,
        highly detailed brushwork,
        masterpiece
    ''',

    "fantasy": '''
        Portrait of a wise old wizard,
        fantasy art style,
        magical glowing effects,
        mystical atmosphere,
        trending on artstation,
        digital painting
    '''
}

# 5. 네거티브 프롬프트
negative_prompt = '''
    low quality, blurry, artifacts,
    deformed hands, bad anatomy,
    poorly drawn face, ugly,
    watermark, text
'''

# 6. 이미지 생성 함수
def generate_image(prompt, negative, seed=42):
    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512
    ).images[0]

    return image

# 7. 여러 스타일 비교
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (style, prompt) in enumerate(prompts.items()):
    print(f"\\n생성 중: {style}...")
    image = generate_image(prompt, negative_prompt)

    axes[idx].imshow(image)
    axes[idx].set_title(f"{style.upper()}", fontsize=14)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('prompt_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n✅ 이미지 생성 완료!")
print("📁 저장 위치: prompt_comparison.png")

# 8. 실험: 네거티브 프롬프트 효과 비교
print("\\n실험: 네거티브 프롬프트 효과...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 네거티브 없음
image_no_neg = generate_image(prompts["realistic"], "", seed=42)
axes[0].imshow(image_no_neg)
axes[0].set_title("네거티브 프롬프트 없음")
axes[0].axis('off')

# 네거티브 있음
image_with_neg = generate_image(prompts["realistic"], negative_prompt, seed=42)
axes[1].imshow(image_with_neg)
axes[1].set_title("네거티브 프롬프트 적용")
axes[1].axis('off')

plt.tight_layout()
plt.savefig('negative_prompt_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n✅ 비교 완료!")
            """, language='python')

        st.markdown('---')

        # 실습 과제
        st.subheader('📝 실습 과제')

        st.warning("""
        **과제**: 효과적인 프롬프트 개발

        1. **3가지 스타일 비교**
           - Realistic, Artistic, Fantasy 스타일로 동일 주제 생성
           - 시드 고정 (재현성 확보)

        2. **네거티브 프롬프트 효과 측정**
           - 동일 프롬프트에 네거티브 있음/없음 비교
           - 품질 개선 정도 분석

        3. **프롬프트 최적화**
           - 5가지 프롬프트 변형 테스트
           - 최적 조합 문서화

        **제출물**: 생성 이미지 + 프롬프트 + 분석 보고서
        """)

    def render_scheduler_lab(self):
        st.header('⏱️ 스케줄러 비교 실습')

        st.markdown("""
        ### 🎯 학습 목표
        - 다양한 스케줄러의 특성 이해
        - 스케줄러별 품질과 속도 비교
        - 용도에 맞는 스케줄러 선택하기
        """)

        st.markdown('---')

        # 스케줄러 비교표
        st.subheader('📊 스케줄러 특성 비교')

        import pandas as pd
        scheduler_data = pd.DataFrame({
            '스케줄러': ['DDPM', 'DDIM', 'DPM-Solver++', 'Euler', 'Euler Ancestral'],
            '권장 스텝': ['1000', '20-50', '20-30', '20-40', '30-40'],
            '속도': ['매우 느림', '빠름', '매우 빠름', '빠름', '중간'],
            '품질': ['최고', '높음', '매우 높음', '높음', '높음'],
            '재현성': ['✅', '✅', '✅', '✅', '❌ (확률적)'],
            '특징': ['원본, 많은 스텝', '적은 스텝 최적화', '최신 기본값', '샤프한 디테일', '창의적 변화']
        })
        st.dataframe(scheduler_data, use_container_width=True, hide_index=True)

        st.markdown('---')

        # 스케줄러별 상세 설명
        st.subheader('🔬 스케줄러 상세 설명')

        sched_tabs = st.tabs(['DDIM', 'DPM-Solver++', 'Euler', 'Euler Ancestral'])

        with sched_tabs[0]:
            st.markdown("""
            ### DDIM (Denoising Diffusion Implicit Models)

            **특징**:
            - 적은 스텝에서도 높은 품질
            - 결정론적 (deterministic) 샘플링
            - 빠른 추론 속도

            **장점**:
            - 20-50 스텝으로 충분
            - 동일 시드로 재현 가능
            - 안정적인 결과

            **단점**:
            - DDPM보다는 품질이 약간 낮음
            - 매우 적은 스텝(<20)에서는 아티팩트 발생

            **추천 용도**: 빠른 프로토타이핑, 실시간 생성
            """)

        with sched_tabs[1]:
            st.markdown("""
            ### DPM-Solver++ (최신 기본값)

            **특징**:
            - ODE 솔버 기반
            - 품질과 속도의 최적 균형
            - Stable Diffusion의 기본 스케줄러

            **장점**:
            - 20-30 스텝으로 최고 품질
            - 매우 빠른 수렴
            - 다양한 모델에서 안정적

            **단점**:
            - 특정 스타일에서는 다른 스케줄러가 더 나을 수 있음

            **추천 용도**: 일반적인 모든 용도, 프로덕션 환경
            """)

        with sched_tabs[2]:
            st.markdown("""
            ### Euler

            **특징**:
            - 오일러 방법 기반
            - 샤프하고 선명한 디테일
            - 단순하고 효율적

            **장점**:
            - 빠른 수렴
            - 샤프한 결과
            - 구현이 단순

            **단점**:
            - 때로는 과도하게 샤프할 수 있음
            - 특정 스타일에 편향 가능

            **추천 용도**: 선명한 디테일이 중요한 경우
            """)

        with sched_tabs[3]:
            st.markdown("""
            ### Euler Ancestral (확률적)

            **특징**:
            - 확률적 (stochastic) 샘플링
            - 매 스텝마다 노이즈 추가
            - 창의적이고 다양한 결과

            **장점**:
            - 높은 다양성
            - 창의적인 변화
            - 예상치 못한 좋은 결과

            **단점**:
            - 재현 불가 (동일 시드도 다른 결과)
            - 불안정할 수 있음
            - 더 많은 스텝 필요

            **추천 용도**: 실험적 생성, 다양성이 중요한 경우
            """)

        st.markdown('---')

        # Google Colab 코드
        st.subheader('💻 Google Colab 실습 코드')

        st.info('🔗 **권장**: 아래 코드를 Google Colab에 복사하여 무료 GPU로 실행하세요!')

        with st.expander('📋 전체 Colab 코드 보기'):
            st.code("""
# ========================================
# Week 9: 스케줄러 비교 실습
# Google Colab에서 실행하세요!
# ========================================

# 1. 패키지 설치
!pip install -q diffusers transformers accelerate torch

# 2. 라이브러리 임포트
import torch
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)
from PIL import Image
import matplotlib.pyplot as plt
import time

# 3. 모델 로드
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 4. 프롬프트 설정
prompt = '''
A serene mountain landscape at sunset,
golden hour lighting,
photorealistic style,
ultra-detailed, 8k resolution
'''

negative_prompt = '''
low quality, blurry, artifacts,
distorted, ugly
'''

# 5. 스케줄러 딕셔너리
schedulers = {
    'DDIM': DDIMScheduler.from_config(pipe.scheduler.config),
    'DPM-Solver++': DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    'Euler': EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    'Euler A': EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
}

# 6. 이미지 생성 함수
def generate_with_scheduler(scheduler_name, scheduler, steps=25):
    pipe.scheduler = scheduler
    generator = torch.Generator("cuda").manual_seed(42)

    start_time = time.time()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512
    ).images[0]

    elapsed_time = time.time() - start_time

    return image, elapsed_time

# 7. 스케줄러 비교 실험
print("스케줄러 비교 실험 시작...")
print("=" * 50)

results = {}
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, (name, scheduler) in enumerate(schedulers.items()):
    print(f"\\n{idx+1}. {name} 생성 중...")

    image, elapsed = generate_with_scheduler(name, scheduler)
    results[name] = {'image': image, 'time': elapsed}

    # 시각화
    axes[idx].imshow(image)
    axes[idx].set_title(f"{name}\\n생성 시간: {elapsed:.2f}초", fontsize=12)
    axes[idx].axis('off')

    print(f"✅ {name}: {elapsed:.2f}초")

plt.tight_layout()
plt.savefig('scheduler_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n" + "=" * 50)
print("✅ 스케줄러 비교 완료!")
print("📁 저장 위치: scheduler_comparison.png")

# 8. 결과 요약
print("\\n📊 결과 요약:")
for name, result in results.items():
    print(f"  - {name:15s}: {result['time']:.2f}초")

# 9. 스텝 수에 따른 비교 (옵션)
print("\\n\\n실험: 스텝 수 영향 분석...")
print("=" * 50)

steps_to_test = [15, 25, 35]
scheduler_to_test = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, steps in enumerate(steps_to_test):
    print(f"\\n{steps} 스텝 생성 중...")
    image, elapsed = generate_with_scheduler("DPM-Solver++", scheduler_to_test, steps)

    axes[idx].imshow(image)
    axes[idx].set_title(f"{steps} Steps\\n{elapsed:.2f}초", fontsize=12)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('steps_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n✅ 스텝 비교 완료!")
            """, language='python')

        st.markdown('---')

        # 실습 가이드
        st.subheader('📝 실습 가이드')

        guide_cols = st.columns(2)

        with guide_cols[0]:
            st.success("""
            **실습 1: 스케줄러 비교**

            1. 동일 프롬프트/시드 사용
            2. 4가지 스케줄러로 생성
            3. 품질과 속도 비교
            4. 스크린샷 저장

            **비교 항목**:
            - 디테일 선명도
            - 색감 및 대비
            - 생성 시간
            - 전체적인 품질
            """)

        with guide_cols[1]:
            st.success("""
            **실습 2: 스텝 수 영향**

            1. 하나의 스케줄러 선택
            2. 15, 25, 35 스텝으로 테스트
            3. 품질 개선 정도 측정
            4. 최적 스텝 수 결정

            **분석 항목**:
            - 스텝별 품질 차이
            - 시간 대비 품질 효율
            - 최소 필요 스텝 수
            """)

        st.warning("""
        **과제**: 스케줄러 최적화 보고서

        1. **4가지 스케줄러 비교** (동일 조건)
        2. **스텝 수 최적화** (15/25/35 스텝)
        3. **용도별 추천** (속도 우선 vs 품질 우선)

        **제출물**: 비교 이미지 + 시간 측정 + 추천 가이드
        """)

    def render_controlnet_lab(self):
        st.header('🧩 ControlNet 실습')

        st.markdown("""
        ### 🎯 학습 목표
        - ControlNet의 원리와 활용법 이해
        - 다양한 조건 신호 활용하기
        - 구도를 제어한 이미지 생성
        """)

        st.markdown('---')

        # ControlNet 타입 비교
        st.subheader('🎨 ControlNet 타입')

        cn_cols = st.columns(4)

        with cn_cols[0]:
            st.markdown("""
            **Canny Edge**
            - 윤곽선 기반
            - 명확한 형태
            - 스케치 → 이미지
            """)

        with cn_cols[1]:
            st.markdown("""
            **OpenPose**
            - 포즈 기반
            - 인체 자세 제어
            - 캐릭터 생성
            """)

        with cn_cols[2]:
            st.markdown("""
            **Depth**
            - 깊이 정보
            - 3D 구조 보존
            - 공간감 제어
            """)

        with cn_cols[3]:
            st.markdown("""
            **Scribble**
            - 낙서 기반
            - 빠른 스케치
            - 자유로운 표현
            """)

        st.markdown('---')

        # Google Colab 코드
        st.subheader('💻 Google Colab 실습 코드')

        st.info('🔗 **권장**: 아래 코드를 Google Colab에 복사하여 실행하세요!')

        with st.expander('📋 Colab 코드: Canny Edge ControlNet'):
            st.code("""
# Week 9: ControlNet 실습 (Canny Edge)

!pip install -q diffusers transformers accelerate torch opencv-python

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. ControlNet 모델 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 2. 입력 이미지에서 Canny edge 추출
def get_canny_edge(image, low_threshold=100, high_threshold=200):
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(edges)

# 3. 샘플 이미지 로드 (URL 또는 로컬 파일)
from urllib.request import urlopen
url = "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
image = Image.open(urlopen(url))

# Canny edge 추출
canny_image = get_canny_edge(image)

# 4. ControlNet으로 이미지 생성
prompt = "a beautiful bird, digital art style, colorful, detailed"
negative_prompt = "low quality, blurry, ugly"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=canny_image,
    num_inference_steps=30,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0
).images[0]

# 5. 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(canny_image)
axes[1].set_title("Canny Edge")
axes[1].axis('off')

axes[2].imshow(output)
axes[2].set_title("ControlNet Output")
axes[2].axis('off')

plt.tight_layout()
plt.savefig('controlnet_result.png', dpi=300)
plt.show()

print("✅ ControlNet 생성 완료!")
            """, language='python')

        st.markdown('---')

        st.warning("""
        **실습 과제**: ControlNet으로 구도 제어

        1. **Canny Edge 실험**: 입력 이미지의 윤곽선 추출 및 생성
        2. **조건 강도 조절**: `controlnet_conditioning_scale` 0.5, 1.0, 1.5 비교
        3. **스타일 변경**: 동일 구도에 다른 스타일 적용

        **제출물**: 입력 이미지 + Canny edge + 생성 결과 3가지
        """)

    def render_comfyui_guide(self):
        st.header('🗺️ ComfyUI 워크플로우 가이드')

        st.markdown("""
        ### 🎯 학습 목표
        - ComfyUI의 노드 시스템 이해
        - 기본 워크플로우 구성
        - 재현 가능한 파이프라인 만들기
        """)

        st.markdown('---')

        # ComfyUI 장점
        st.subheader('✨ ComfyUI 장점')

        adv_cols = st.columns(4)

        with adv_cols[0]:
            st.markdown("""
            **🔧 시각적 편집**
            - 노드 기반
            - 직관적 인터페이스
            - 드래그 & 드롭
            """)

        with adv_cols[1]:
            st.markdown("""
            **🔄 재현성**
            - 워크플로우 저장
            - JSON 공유
            - 버전 관리
            """)

        with adv_cols[2]:
            st.markdown("""
            **🎯 유연성**
            - 복잡한 파이프라인
            - 커스텀 노드
            - 모듈식 구성
            """)

        with adv_cols[3]:
            st.markdown("""
            **📊 협업**
            - 팀 공유
            - 표준화
            - 문서화 용이
            """)

        st.markdown('---')

        # 기본 워크플로우
        st.subheader('🔨 기본 워크플로우')

        st.code("""
[Load Checkpoint]
     ↓
[CLIP Text Encode (Positive)]
     ↓
[CLIP Text Encode (Negative)]
     ↓
[Empty Latent Image]
     ↓
[KSampler]
     ↓
[VAE Decode]
     ↓
[Save Image]
        """, language='text')

        st.markdown('---')

        # 설치 가이드
        st.subheader('💻 ComfyUI 설치')

        install_tabs = st.tabs(['Windows', 'Mac/Linux', 'Google Colab'])

        with install_tabs[0]:
            st.code("""
# Windows 설치

# 1. Git 클론
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. 가상환경 생성
python -m venv venv
venv\\Scripts\\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 실행
python main.py

# 5. 브라우저에서 접속
# http://127.0.0.1:8188
            """, language='bash')

        with install_tabs[1]:
            st.code("""
# Mac/Linux 설치

# 1. Git 클론
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 실행
python main.py

# 5. 브라우저에서 접속
# http://127.0.0.1:8188
            """, language='bash')

        with install_tabs[2]:
            st.code("""
# Google Colab 설치

# 1. 설치 및 실행
!git clone https://github.com/comfyanonymous/ComfyUI.git
%cd ComfyUI
!pip install -q -r requirements.txt

# 2. Colab에서 실행 (ngrok 터널)
!pip install pyngrok
from pyngrok import ngrok

# ngrok 터널 시작
public_url = ngrok.connect(8188)
print(f"ComfyUI URL: {public_url}")

# ComfyUI 실행
!python main.py --listen 0.0.0.0 --port 8188
            """, language='python')

        st.markdown('---')

        # 실습 가이드
        st.subheader('📝 실습 가이드')

        st.success("""
        **기본 워크플로우 실습**

        **1. 모델 로드**
        - `Load Checkpoint` 노드 추가
        - Stable Diffusion v1.5 선택

        **2. 텍스트 입력**
        - `CLIP Text Encode` 노드 2개 (positive, negative)
        - 프롬프트 입력

        **3. 샘플링 설정**
        - `Empty Latent Image` (512x512)
        - `KSampler` (steps: 25, cfg: 7.5, seed: 42)

        **4. 이미지 디코딩**
        - `VAE Decode` 노드
        - `Save Image` 노드

        **5. 노드 연결 및 실행**
        - 모든 노드를 순서대로 연결
        - Queue Prompt 버튼 클릭
        """)

        st.markdown('---')

        # 고급 워크플로우
        st.subheader('🚀 고급 워크플로우')

        adv_workflow_tabs = st.tabs(['LoRA 적용', 'ControlNet', '배치 생성'])

        with adv_workflow_tabs[0]:
            st.markdown("""
            **LoRA 워크플로우**
            ```
            [Load Checkpoint]
                 ↓
            [Load LoRA]
                 ↓
            [CLIP Text Encode]
                 ↓
            [KSampler]
                 ↓
            [VAE Decode]
                 ↓
            [Save Image]
            ```

            **추가 노드**:
            - `Load LoRA`: LoRA 가중치 로드
            - `LoRA Stack`: 여러 LoRA 조합
            """)

        with adv_workflow_tabs[1]:
            st.markdown("""
            **ControlNet 워크플로우**
            ```
            [Load Checkpoint]
                 ↓
            [Load ControlNet Model]
                 ↓
            [Load Image] → [ControlNet Preprocessor]
                 ↓
            [Apply ControlNet]
                 ↓
            [CLIP Text Encode]
                 ↓
            [KSampler]
                 ↓
            [VAE Decode]
                 ↓
            [Save Image]
            ```

            **주요 노드**:
            - `ControlNet Preprocessor`: 조건 신호 추출
            - `Apply ControlNet`: ControlNet 적용
            """)

        with adv_workflow_tabs[2]:
            st.markdown("""
            **배치 생성 워크플로우**
            ```
            [Load Checkpoint]
                 ↓
            [CLIP Text Encode (Multiple)]
                 ↓
            [Empty Latent Image Batch]
                 ↓
            [KSampler] (batch_size: N)
                 ↓
            [VAE Decode]
                 ↓
            [Save Image] (각각 저장)
            ```

            **특징**:
            - 여러 이미지 동시 생성
            - 다른 시드 자동 적용
            - 효율적인 배치 처리
            """)

        st.markdown('---')

        st.warning("""
        **실습 과제**: ComfyUI 워크플로우 구성

        1. **기본 워크플로우 구성**
           - Text-to-Image 파이프라인 구축
           - 동일 결과 재현 (시드 고정)

        2. **워크플로우 저장 및 공유**
           - JSON 파일 저장
           - 팀원과 공유

        3. **3가지 스타일 이미지 생성**
           - 동일 워크플로우
           - 프롬프트만 변경
           - 결과 비교

        **제출물**: 워크플로우 JSON + 생성 이미지 3장 + 스크린샷
        """)


