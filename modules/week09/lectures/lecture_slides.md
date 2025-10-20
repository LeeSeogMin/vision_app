# Week 9: 생성 모델 이론 + Stable Diffusion

## 강의 개요

**주제**: Text-to-Image 생성 모델의 이론과 실습

**학습 목표**:
- Diffusion 모델의 원리 이해
- Stable Diffusion 파이프라인 구성 요소 파악
- 효과적인 프롬프트 엔지니어링 기법 습득
- ControlNet을 활용한 조건부 생성
- ComfyUI 워크플로우 구성

---

## 1. 생성 모델 개론

### 1.1 생성 모델의 발전

```
GAN (2014)
  ↓
VAE (Variational Autoencoder)
  ↓
Diffusion Models (2020)
  ↓
Stable Diffusion (2022)
```

### 1.2 주요 생성 모델 비교

| 모델 | 장점 | 단점 | 대표 사례 |
|------|------|------|-----------|
| **GAN** | 빠른 생성 | 학습 불안정 | StyleGAN, DALL-E 1 |
| **VAE** | 안정적 학습 | 흐릿한 결과 | Variational AE |
| **Diffusion** | 고품질 결과 | 느린 생성 | Stable Diffusion, Imagen |

---

## 2. Diffusion 모델의 원리

### 2.1 Forward Process (순방향 과정)

```
원본 이미지 → 노이즈 추가 → 노이즈 추가 → ... → 순수 노이즈
x₀          x₁              x₂              ...   xₜ
```

**수식**:
```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)
```

### 2.2 Reverse Process (역방향 과정)

```
순수 노이즈 → 노이즈 제거 → 노이즈 제거 → ... → 원본 이미지
xₜ           xₜ₋₁            xₜ₋₂            ...   x₀
```

**학습 목표**: 노이즈를 예측하는 신경망 학습
```
ε_θ(xₜ, t) ≈ ε (실제 노이즈)
```

### 2.3 손실 함수

```
L = E[||ε - ε_θ(xₜ, t)||²]
```

---

## 3. Stable Diffusion 아키텍처

### 3.1 전체 파이프라인

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

### 3.2 주요 구성 요소

#### 3.2.1 VAE (Variational Autoencoder)

**목적**: 고차원 이미지 → 저차원 잠재 공간

```
Encoder: 512×512 이미지 → 64×64 잠재 표현 (8배 압축)
Decoder: 64×64 잠재 표현 → 512×512 이미지
```

**장점**:
- 메모리 효율성 (64배 감소)
- 빠른 샘플링
- 고품질 재구성

#### 3.2.2 CLIP Text Encoder

**목적**: 텍스트 → 의미 벡터

```
"A beautiful sunset" → [0.23, -0.45, 0.67, ...]
                       (77 토큰 × 768 차원)
```

#### 3.2.3 U-Net

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
```
Query: 이미지 특징 (U-Net)
Key/Value: 텍스트 임베딩 (CLIP)
→ 텍스트-이미지 정렬
```

---

## 4. 샘플링 스케줄러

### 4.1 스케줄러의 역할

**목적**: 노이즈 제거 과정의 타임스텝 관리

### 4.2 주요 스케줄러 비교

#### 4.2.1 DDPM (Denoising Diffusion Probabilistic Models)

```python
# 전통적 방법, 많은 스텝 필요
steps = 1000  # 느림
quality = "High"
```

#### 4.2.2 DDIM (Denoising Diffusion Implicit Models)

```python
# 빠른 샘플링
steps = 20-50  # 빠름
quality = "High"
deterministic = True  # 재현 가능
```

#### 4.2.3 DPM-Solver++

```python
# 최신 기본값
steps = 20-30
quality = "Very High"
speed = "Fast"
```

#### 4.2.4 Euler / Euler Ancestral

```python
# Euler: 안정적, 샤프
# Euler A: 창의적, 변화 큼
steps = 20-40
```

### 4.3 스케줄러 선택 가이드

| 용도 | 추천 스케줄러 | 스텝 수 |
|------|---------------|---------|
| 일반 생성 | DPM-Solver++ | 25-30 |
| 빠른 프로토타이핑 | DDIM | 20-25 |
| 고품질 | DPM-Solver++ | 40-50 |
| 실험적 | Euler Ancestral | 30-40 |

---

## 5. 프롬프트 엔지니어링

### 5.1 효과적인 프롬프트 구조

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

### 5.2 스타일 토큰

#### 예술 스타일
```
- "oil painting"
- "watercolor"
- "digital art"
- "concept art"
- "photorealistic"
```

#### 품질 향상
```
- "highly detailed"
- "ultra-detailed"
- "8k resolution"
- "sharp focus"
- "professional"
```

#### 조명
```
- "cinematic lighting"
- "studio lighting"
- "natural lighting"
- "dramatic lighting"
- "soft lighting"
```

### 5.3 네거티브 프롬프트

**목적**: 원하지 않는 요소 제거

```python
negative_prompt = """
low quality, blurry, artifacts,
deformed hands, extra fingers,
bad anatomy, poorly drawn face,
mutation, distorted
"""
```

### 5.4 가중치 조정

```python
# 강조 (가중치 증가)
"(beautiful landscape:1.3)"  # 1.3배 강조

# 약화 (가중치 감소)
"(people:0.7)"  # 0.7배 약화
```

---

## 6. 파라미터 최적화

### 6.1 주요 파라미터

#### 6.1.1 CFG Scale (Classifier-Free Guidance)

```python
guidance_scale = 7.5  # 기본값

# 낮은 값 (1-5): 창의적, 프롬프트 이탈
# 중간 값 (7-9): 균형잡힌 결과
# 높은 값 (10-15): 프롬프트 충실, 과포화
```

#### 6.1.2 Inference Steps

```python
num_inference_steps = 25  # 기본값

# 적은 스텝 (10-20): 빠름, 품질 저하
# 중간 스텝 (25-35): 균형
# 많은 스텝 (40-50): 느림, 품질 향상 (미미)
```

#### 6.1.3 Seed

```python
generator = torch.Generator().manual_seed(42)
# 동일 시드 = 재현 가능한 결과
```

### 6.2 파라미터 조합 가이드

| 목적 | Steps | CFG | 추천 스케줄러 |
|------|-------|-----|---------------|
| 빠른 테스트 | 15-20 | 7.0 | DDIM |
| 일반 생성 | 25-30 | 7.5 | DPM-Solver++ |
| 고품질 | 40-50 | 8.0 | DPM-Solver++ |
| 실험적 | 30-40 | 6.0-9.0 | Euler A |

---

## 7. ControlNet

### 7.1 ControlNet이란?

**목적**: 추가 조건 신호로 생성 과정 제어

```
텍스트 프롬프트 + 조건 이미지 → 구도가 제어된 결과
```

### 7.2 주요 ControlNet 타입

#### 7.2.1 Canny Edge

```python
# 윤곽선 기반 제어
use_case = "스케치 → 이미지"
강점 = "명확한 형태 보존"
```

#### 7.2.2 OpenPose

```python
# 포즈 기반 제어
use_case = "인체 자세 제어"
강점 = "캐릭터 포즈 정확도"
```

#### 7.2.3 Depth

```python
# 깊이 정보 기반
use_case = "3D 구조 보존"
강점 = "공간감, 원근감"
```

#### 7.2.4 Scribble

```python
# 간단한 낙서 기반
use_case = "빠른 스케치 → 이미지"
강점 = "자유로운 표현"
```

### 7.3 ControlNet 사용 예시

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# ControlNet 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)

# 파이프라인 구성
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet
)

# 생성
output = pipe(
    prompt="A beautiful landscape",
    image=canny_edge_image,
    num_inference_steps=30,
    controlnet_conditioning_scale=1.0  # 조건 강도
).images[0]
```

---

## 8. 경량 미세튜닝

### 8.1 LoRA (Low-Rank Adaptation)

**목적**: 적은 파라미터로 스타일 학습

```
전체 모델: 4GB
LoRA 가중치: 10-100MB (400배 작음)
```

**장점**:
- 빠른 학습
- 낮은 메모리 요구
- 여러 LoRA 조합 가능

### 8.2 LoRA 적용

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

# LoRA 로드
pipe.load_lora_weights("path/to/lora")

# 가중치 조정
pipe.set_lora_scale(0.8)  # 0.0-1.0
```

---

## 9. ComfyUI

### 9.1 ComfyUI란?

**특징**:
- 🔧 노드 기반 시각적 편집
- 🔄 워크플로우 저장/공유
- 🎯 복잡한 파이프라인 구성
- 📊 팀 협업 용이

### 9.2 기본 워크플로우

```
[Load Checkpoint]
      ↓
[CLIP Text Encode (Positive)]
      ↓
[CLIP Text Encode (Negative)]
      ↓
[KSampler]
      ↓
[VAE Decode]
      ↓
[Save Image]
```

### 9.3 고급 워크플로우

#### ControlNet + LoRA

```
[Load Checkpoint]
      ↓
[Load LoRA]
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

---

## 10. 안전 및 윤리

### 10.1 저작권 고려사항

- ✅ 학습 목적 사용
- ✅ 개인 프로젝트
- ⚠️ 상업적 사용 시 라이선스 확인
- ❌ 저작권 침해 이미지 생성

### 10.2 안전 필터

```python
# Stable Diffusion의 Safety Checker
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=safety_checker,  # 기본 활성화
    requires_safety_checker=True
)
```

### 10.3 편향 및 윤리

**주의사항**:
- 성별, 인종, 연령 편향 인식
- 유해 콘텐츠 생성 금지
- 프라이버시 존중
- 딥페이크 악용 방지

---

## 11. 실습 환경

### 11.1 Google Colab (권장)

```python
# GPU 설정: T4 (무료)
# 런타임 → 런타임 유형 변경 → GPU

# 패키지 설치
!pip install -q diffusers transformers accelerate torch --upgrade

# 모델 다운로드 (약 5GB)
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)
```

### 11.2 로컬 환경

```bash
# venv 활성화
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 패키지 설치
pip install diffusers transformers accelerate torch --upgrade
```

**시스템 요구사항**:
- GPU: 4GB VRAM 이상 권장
- RAM: 8GB 이상
- 저장 공간: 10GB 이상 (모델 포함)

---

## 12. 실습 과제

### 과제 1: 기본 생성

**목표**: Stable Diffusion으로 3가지 스타일 이미지 생성

**요구사항**:
- 동일 프롬프트
- 다른 스타일 토큰 (realistic, anime, oil painting)
- 파라미터 비교 분석

### 과제 2: 프롬프트 최적화

**목표**: 효과적인 프롬프트 개발

**요구사항**:
- 5가지 프롬프트 변형 테스트
- 네거티브 프롬프트 효과 비교
- 최적 조합 문서화

### 과제 3: ControlNet 실습

**목표**: ControlNet으로 구도 제어

**요구사항**:
- Canny edge 추출
- 동일 구도, 다른 스타일 3장
- 결과 비교 분석

---

## 13. 참고 자료

### 공식 문서
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)
- [Stable Diffusion 공식 GitHub](https://github.com/Stability-AI/stablediffusion)

### 논문
- "Denoising Diffusion Probabilistic Models" (DDPM)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
- "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)

### 커뮤니티
- [r/StableDiffusion](https://reddit.com/r/StableDiffusion)
- [Civitai](https://civitai.com) - 모델 및 LoRA 공유

---

## 요약

### 핵심 개념
1. **Diffusion 원리**: 노이즈 추가 → 노이즈 제거 학습
2. **Stable Diffusion**: VAE + U-Net + CLIP
3. **프롬프트 엔지니어링**: 구조화된 프롬프트 + 네거티브
4. **스케줄러**: 샘플링 과정 최적화
5. **ControlNet**: 조건부 생성
6. **LoRA**: 경량 스타일 미세튜닝

### 다음 주 예고
**Week 10**: 실전 프로젝트 및 종합 실습
- 멀티모달 AI 애플리케이션 개발
- 전체 모듈 통합
- 최종 프로젝트 발표

---

**질문 & 토론**

Q&A 시간입니다. 궁금한 점이 있으시면 언제든 질문해주세요! 🙋‍♂️🙋‍♀️
