"""
Week 10: 자율주행 인식 시스템 (End-to-End Autonomous Driving Pipeline)

이 모듈은 완전한 자율주행 인식 시스템을 다룹니다:
- SAE 자율주행 레벨 0-5 이해
- 차선 인식: 직선(Hough) → 곡선(Polynomial) → 딥러닝(LaneNet)
- 객체 탐지 및 추적: YOLOv8 + ByteTrack + 거리 추정
- 통합 파이프라인: 차선 + 객체 + 위험도 분석
- 3D 시각화: BEV (Bird's Eye View) + 3D 바운딩 박스
- 의사결정 시스템: 차선이탈/충돌/급정거 대응
- 고급 시뮬레이터: 교차로, 신호등, 날씨 효과
- 실전 배포: TensorRT 최적화, Edge 디바이스
"""

from .autonomous_driving_module import AutonomousDrivingModule

__all__ = ['AutonomousDrivingModule']
