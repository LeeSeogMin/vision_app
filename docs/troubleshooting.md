# Troubleshooting Guide

## 이미지 로딩 오류 (404 Error)

### 증상
```
Failed to load resource: the server responded with a status of 404 (Not Found)
Client Error: Image source error - http://localhost:8501/media/xxxxx.jpg
```

### 원인
- Streamlit의 미디어 캐시가 세션 간 동기화되지 않음
- 이전 세션의 이미지 해시가 현재 세션에서 무효화됨

### 해결 방법

#### 방법 1: 캐시 초기화 (빠른 해결)

**브라우저에서**:
1. `C` 키 - 캐시 클리어
2. `R` 키 - 앱 새로고침

**터미널에서**:
```bash
# 앱 중지 (Ctrl+C)
# 캐시 정리
rm -rf .streamlit/cache ~/.streamlit/cache

# 앱 재시작
streamlit run app.py
```

#### 방법 2: 이미지 로딩 방식 개선

**현재 방식** (캐시 문제 발생):
```python
image = Image.open(uploaded_file)
st.image(image)  # 캐시 해시 생성
```

**권장 방식** (캐시 안전):
```python
import io
from PIL import Image

# 바이트 버퍼로 직접 처리
if uploaded_file is not None:
    # 파일 포인터 초기화
    uploaded_file.seek(0)

    # 이미지 로드
    image = Image.open(uploaded_file)

    # RGB 변환 (일관성)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 표시
    st.image(image, use_container_width=True)
```

#### 방법 3: Streamlit 설정 최적화

**`.streamlit/config.toml` 생성**:
```toml
[server]
# 파일 업로드 크기 제한 증가
maxUploadSize = 200

# 메시지 크기 제한 증가
maxMessageSize = 200

[browser]
# 서버 포트 고정
serverPort = 8501

# 자동 새로고침 비활성화 (안정성)
gatherUsageStats = false
```

### 예방 조치

1. **파일 포인터 관리**:
```python
# 업로드 파일 사용 전 항상 초기화
uploaded_file.seek(0)
```

2. **이미지 형식 통일**:
```python
# 일관된 형식 사용
if image.mode != 'RGB':
    image = image.convert('RGB')
```

3. **메모리 정리**:
```python
# 큰 이미지 처리 후 정리
import gc
gc.collect()
```

## 기타 일반적인 오류

### Chrome Extension 오류
```
Uncaught TypeError: Cannot read properties of undefined
```

**원인**: 브라우저 확장 프로그램과 Streamlit 충돌

**해결**:
- 무시 가능 (Streamlit 앱 동작에 영향 없음)
- 또는 시크릿 모드에서 실행

### Vega-Lite 버전 경고
```
The input spec uses Vega-Lite v5.20.1, but the current version is v6.3.1
```

**원인**: 차트 라이브러리 버전 차이

**해결**:
- 무시 가능 (시각화는 정상 동작)
- 업그레이드 원하면: `pip install -U altair`

### Popper.js 경고
```
`preventOverflow` modifier is required by `hide` modifier
```

**원인**: Streamlit 내부 라이브러리 경고

**해결**:
- 무시 가능 (UI 동작에 영향 없음)

## 성능 최적화

### 메모리 사용량 최적화

```python
import streamlit as st
from PIL import Image

@st.cache_resource
def load_large_model():
    # 모델은 한 번만 로드
    return model

@st.cache_data
def process_image(_image):
    # 이미지 처리 결과 캐싱
    # 주의: PIL Image는 언더스코어로 시작하는 인자명 사용
    return processed
```

### 세션 상태 관리

```python
# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.results = []

# 데이터 누적 방지
if st.button('분석 시작'):
    # 이전 결과 초기화
    st.session_state.results = []
```

## 도움이 필요하신가요?

문제가 지속되면:
1. GitHub Issues 확인
2. Streamlit 커뮤니티 포럼
3. 프로젝트 관리자에게 문의
