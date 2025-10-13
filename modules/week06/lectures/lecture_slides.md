# Week 6: 이미지 세그멘테이션과 SAM

## 📋 강의 개요

**학습 목표**:
- 세그멘테이션의 종류와 차이점 이해
- U-Net 아키텍처의 핵심 구조 학습
- Segment Anything Model (SAM) 원리와 활용법 습득
- 실전 응용: 배경 제거, 자동 라벨링 구현

**강의 시간**: 180분 (이론 90분 + 실습 90분)

---

## Part 1: 세그멘테이션 기초 (30분)

### 1.1 세그멘테이션이란?

**정의**: 이미지의 각 픽셀을 특정 클래스나 객체에 할당하는 작업

**Computer Vision 태스크 비교**:
| 태스크 | 입력 | 출력 | 정밀도 |
|--------|------|------|--------|
| Classification | 이미지 | 클래스 레이블 | 이미지 전체 |
| Object Detection | 이미지 | BBox + 클래스 | 박스 단위 |
| Segmentation | 이미지 | 픽셀별 마스크 | 픽셀 단위 |

### 1.2 세그멘테이션의 종류

#### Semantic Segmentation
- **목표**: 같은 클래스의 모든 픽셀에 같은 레이블
- **특징**: 개별 인스턴스 구분 없음
- **예시**:
  - "road" 클래스 → 모든 도로 픽셀
  - "sky" 클래스 → 모든 하늘 픽셀
- **응용**: 자율주행 (차선, 도로, 보행자 구역)

#### Instance Segmentation
- **목표**: 같은 클래스라도 개별 객체 구분
- **특징**: 각 인스턴스마다 별도 마스크
- **예시**:
  - person_1, person_2, person_3
  - car_1, car_2, car_3
- **응용**: 객체 추적, 로봇 비전

#### Panoptic Segmentation
- **목표**: Semantic + Instance 결합
- **구분**:
  - **Stuff**: 배경 요소 (하늘, 도로) → Semantic
  - **Thing**: 개별 객체 (사람, 차) → Instance
- **응용**: 완전한 장면 이해

### 1.3 세그멘테이션의 난이도

**Challenge 1: 경계 정확도**
- 픽셀 단위 정밀 예측 필요
- 객체 경계가 불분명한 경우 (머리카락, 나뭇잎 등)

**Challenge 2: 다양한 스케일**
- 큰 객체와 작은 객체 동시 처리
- Multi-scale feature 필요

**Challenge 3: Occlusion**
- 가려진 객체 처리
- 겹치는 객체 경계 구분

---

## Part 2: U-Net 아키텍처 (30분)

### 2.1 U-Net 역사

**배경**:
- 2015년 Ronneberger et al. 발표
- 의료 이미지 세그멘테이션을 위해 개발
- 적은 데이터로 높은 성능

**혁신성**:
- Skip connections로 세부 정보 보존
- Encoder-Decoder 구조
- Data augmentation 활용

### 2.2 U-Net 구조

```
입력 이미지 (572x572x3)
    ↓
┌─────────────────────────────────────┐
│   Contracting Path (Encoder)        │
│                                     │
│   Conv 3x3 + ReLU (2번)            │
│   ↓ Max Pool 2x2                    │
│   64 → 128 → 256 → 512 → 1024      │
│   (공간 해상도 감소, 채널 증가)         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Bottleneck (1024 channels)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Expansive Path (Decoder)          │
│                                     │
│   Up-Conv 2x2                       │
│   + Skip Connection                 │
│   + Conv 3x3 + ReLU (2번)          │
│   1024 → 512 → 256 → 128 → 64      │
│   (공간 해상도 복원, 채널 감소)         │
└─────────────────────────────────────┘
    ↓
출력 마스크 (388x388x2)
```

### 2.3 핵심 구성 요소

#### Skip Connections ⭐
```
Encoder Level 1 ─────────→ Concat → Decoder Level 4
Encoder Level 2 ─────────→ Concat → Decoder Level 3
Encoder Level 3 ─────────→ Concat → Decoder Level 2
Encoder Level 4 ─────────→ Concat → Decoder Level 1
```

**역할**:
1. **세부 정보 보존**: 고해상도 특징을 직접 전달
2. **Gradient Flow**: 학습 안정성 향상
3. **위치 정보**: 정확한 픽셀 위치 복원

#### Contracting Path (Encoder)
```python
# Pseudo code
for level in [1, 2, 3, 4]:
    x = Conv3x3_ReLU(x)  # 특징 추출
    x = Conv3x3_ReLU(x)
    skip_connections[level] = x  # 저장
    x = MaxPool2x2(x)  # 다운샘플링
```

#### Expansive Path (Decoder)
```python
# Pseudo code
for level in [4, 3, 2, 1]:
    x = UpConv2x2(x)  # 업샘플링
    x = Concat(x, skip_connections[level])  # Skip connection
    x = Conv3x3_ReLU(x)
    x = Conv3x3_ReLU(x)
```

### 2.4 U-Net 변형

- **U-Net++**: Nested skip pathways
- **Attention U-Net**: Attention gates 추가
- **3D U-Net**: 3D 의료 영상용
- **ResUNet**: Residual connections

---

## Part 3: Segment Anything Model (SAM) (30분)

### 3.1 SAM 소개

**개발**: Meta AI (2023년 4월)

**혁신성**:
- **Zero-shot Transfer**: 학습되지 않은 객체도 분할
- **Promptable**: Point, Box, Mask로 제어
- **대규모 데이터**: SA-1B (11M 이미지, 1.1B 마스크)
- **범용성**: 다양한 도메인에 적용 가능

**기존 모델과의 차이**:
| 특징 | 기존 모델 | SAM |
|------|----------|-----|
| 학습 데이터 | 특정 도메인 | 범용 (11M 이미지) |
| 분할 대상 | 고정된 클래스 | 임의의 객체 |
| 프롬프트 | 불가능 | Point, Box, Mask |
| Zero-shot | Fine-tuning 필요 | 즉시 가능 |

### 3.2 SAM 아키텍처

#### 전체 구조
```
┌─────────────────────────────────────────┐
│  Image Encoder (ViT-H/L/B)              │
│  - Vision Transformer                   │
│  - 1024x1024 → 64x64x256 embedding     │
│  - Heavy, 한 번만 실행                    │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Prompt Encoder                         │
│  - Point: Positional encoding           │
│  - Box: Corner embeddings               │
│  - Mask: Conv downsampling              │
│  - Lightweight                          │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  Mask Decoder                           │
│  - Transformer decoder (2 layers)       │
│  - 3개 마스크 출력 (ambiguity)            │
│  - IoU confidence score                 │
│  - Lightweight, 실시간 가능               │
└─────────────────────────────────────────┘
```

#### Image Encoder: Vision Transformer
- **ViT-B**: ~375MB, 12 layers
- **ViT-L**: ~1.2GB, 24 layers
- **ViT-H**: ~2.4GB, 32 layers

**특징**:
- MAE (Masked Autoencoder) pre-training
- Global receptive field
- 고해상도 특징 추출

#### Prompt Encoder
```python
# Point Prompt
point_embedding = positional_encoding(x, y) + fg/bg_token

# Box Prompt
box_embedding = corner_embeddings(x1, y1, x2, y2)

# Mask Prompt
mask_embedding = conv_layers(sparse_mask)
```

#### Mask Decoder
- Transformer 기반 디코더 (경량)
- 3개 후보 마스크 출력
  - **이유**: Ambiguity 처리 (다양한 해석 가능)
  - 예: "컵" 클릭 → 컵만? 컵+손잡이? 전체 테이블?
- IoU prediction head로 품질 평가

### 3.3 SAM의 학습 전략

#### Data Engine (3단계)
1. **Assisted-manual stage**:
   - 전문 annotator가 SAM 도움으로 라벨링
   - 4.3M 마스크 수집

2. **Semi-automatic stage**:
   - SAM이 자동 제안 → 사람이 검증
   - 10.2M 마스크 추가

3. **Fully automatic stage**:
   - SAM이 완전 자동으로 생성
   - 1.1B 마스크 (최종 데이터셋)

#### Loss Function
```
Total Loss = Focal Loss + Dice Loss + IoU Loss

- Focal Loss: 클래스 불균형 해결
- Dice Loss: 세그멘테이션 품질
- IoU Loss: IoU 예측 정확도
```

### 3.4 SAM 활용 방법

#### Point Prompt
```python
# Foreground point (x, y, label=1)
sam.segment_with_points(
    image,
    points=[(300, 200)],
    labels=[1]
)

# Multi-point (fg + bg)
sam.segment_with_points(
    image,
    points=[(300, 200), (100, 100)],
    labels=[1, 0]  # 1=fg, 0=bg
)
```

#### Box Prompt
```python
# Bounding box (x1, y1, x2, y2)
sam.segment_with_box(
    image,
    box=(100, 100, 400, 400)
)
```

#### Automatic Mask Generation
```python
# Grid sampling → multiple masks
masks = sam.generate_auto_masks(
    image,
    points_per_side=32,  # 그리드 밀도
    pred_iou_thresh=0.88,  # 품질 필터
    stability_score_thresh=0.95
)
```

---

## Part 4: 실습 가이드 (90분)

### 실습 1: SAM 기초 사용 (20분)

**목표**: Point/Box 프롬프트로 세그멘테이션

**실습 내용**:
1. 샘플 이미지 로드
2. Point prompt로 객체 분할
3. Background point로 정밀화
4. Box prompt로 영역 지정

**학습 포인트**:
- Foreground vs Background label
- 여러 포인트의 효과
- 박스 프롬프트의 장단점

### 실습 2: Interactive Annotation (25분)

**목표**: 반복적으로 포인트 추가하며 마스크 개선

**워크플로우**:
1. 초기 포인트로 대략 분할
2. 누락 영역에 fg point 추가
3. 과도 포함 영역에 bg point 추가
4. 만족할 때까지 반복

**학습 포인트**:
- Interactive segmentation 프로세스
- 효율적인 annotation 전략
- 실무 데이터 라벨링 시뮬레이션

### 실습 3: Auto Mask Generation (20분)

**목표**: 프롬프트 없이 전체 이미지 자동 분할

**실습 내용**:
1. 자동 마스크 생성
2. 파라미터 조정 (points_per_side)
3. 품질 필터링 (IoU, stability)
4. 결과 시각화 및 분석

**학습 포인트**:
- Grid sampling 원리
- NMS (Non-Maximum Suppression)
- 품질 메트릭의 의미

### 실습 4: 배경 제거 앱 (15분)

**목표**: 증명사진 자동 편집기 구현

**기능**:
1. 인물 사진 업로드
2. SAM으로 인물 분할
3. 배경 색상 변경
4. 결과 다운로드

**응용**: 증명사진, 상품 이미지, 프로필 사진

### 실습 5: 자동 라벨링 도구 (10분)

**목표**: 객체 탐지 학습 데이터 생성 자동화

**워크플로우**:
1. 이미지에서 자동 마스크 생성
2. 각 마스크에 클래스 레이블 할당
3. BBox 추출
4. COCO/YOLO 포맷 변환

**실무 가치**: 데이터 라벨링 시간 90% 감소

---

## Part 5: 심화 주제

### 5.1 SAM의 한계

**현재 한계**:
1. **클래스 인식 불가**: SAM은 "무엇을 분할할지"만 결정, "무엇인지"는 모름
2. **영상 처리**: 동영상은 프레임별 처리 필요 (일관성 부족)
3. **Fine details**: 머리카락, 투명 객체 등 어려움
4. **계산 비용**: ViT-H는 고사양 GPU 필요

**해결 방안**:
- **Grounded-SAM**: CLIP + SAM (텍스트 프롬프트)
- **Track-Anything**: 영상 추적 + SAM
- **HQ-SAM**: 고품질 마스크 생성

### 5.2 SAM 변형 모델들

#### Grounded-SAM
```
Text: "a red car" → CLIP → Object Detection → SAM
```
- 자연어로 객체 지정 가능
- Zero-shot object detection

#### Mobile-SAM
- ViT 대신 경량 encoder
- 모바일 기기에서 실시간 동작
- 성능 약간 하락, 속도 10배 향상

#### HQ-SAM
- High-quality mask 생성
- 미세한 경계 개선
- 추가 학습 필요

### 5.3 실전 배포 고려사항

**성능 최적화**:
```python
# 1. 이미지 크기 제한
image = resize_if_needed(image, max_size=1024)

# 2. 모델 캐싱
@st.cache_resource
def load_sam():
    return SAMHelper()

# 3. Batch processing
masks = sam.batch_predict(images)  # 여러 이미지 동시 처리
```

**메모리 관리**:
- ViT-B 사용 (375MB)
- 이미지 전처리로 크기 제한
- GPU 메모리 부족 시 CPU fallback

**사용자 경험**:
- Progress bar로 진행 상황 표시
- 중간 결과 미리보기
- 오류 발생 시 명확한 메시지

---

## 요약 및 다음 주 예고

### 이번 주 핵심 내용
✅ Semantic, Instance, Panoptic Segmentation 구분
✅ U-Net의 Skip Connections 이해
✅ SAM의 Promptable Segmentation 활용
✅ 실전 응용: 배경 제거, 자동 라벨링

### 다음 주 (Week 7): Action Recognition
- 영상 데이터 처리
- 3D CNN vs 2D CNN + LSTM
- Temporal modeling
- 실시간 행동 인식

---

## 추가 학습 자료

### 논문
- U-Net: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
- SAM: [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)

### 데모
- SAM Demo: [https://segment-anything.com/](https://segment-anything.com/)
- Grounded-SAM: [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

### 코드
- Official SAM: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
- HuggingFace: [https://huggingface.co/docs/transformers/model_doc/sam](https://huggingface.co/docs/transformers/model_doc/sam)

---

**강의 종료 | Week 6 완료 🎉**
