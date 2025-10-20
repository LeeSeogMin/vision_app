"""
Week 9: 생성 모델 이론 + Stable Diffusion (Text-to-Image Generation)

이 모듈은 생성 모델의 이론과 Stable Diffusion을 활용한 텍스트-이미지 생성을 다룹니다:
- VAE와 잠재 공간: 인코더/디코더, 샘플링
- Diffusion 역과정: 노이즈 제거, 스케줄러(DDIM/DPMSolver)
- 파이프라인 구성: 토크나이저 → 텍스트 인코더 → U-Net → VAE
- 프롬프트 엔지니어링: 스타일 토큰, 네거티브 프롬프트
- 스케줄러 비교: DDIM, DPMSolver, Euler
- ControlNet/Adapter: 조건부 생성
- ComfyUI 가이드: 노드 기반 워크플로우
- 경량 미세튜닝: LoRA/LoCon
- 안전/윤리 가이드: 저작권, 안전 필터, 편향
"""

from .generation_module import GenerationModule

__all__ = ['GenerationModule']

