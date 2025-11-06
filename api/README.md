# HnV SafetyEyes Project

수계 및 사람 탐지를 위한 FastAPI 기반 백엔드 서버입니다.

## 기능

- 수계 Segmentation
- 사람 Detection

## 설치 및 실행

### 환경

- python==3.11.9
- Window 10, 11

### 설치

```bash
python -m venv .venv
.venv\Scripts\activate.ps1
pip install -r requirements.txt
```

### 모델 파일 준비

ONNX 모델 파일을 `models` 디렉토리에 복사합니다:

- `models/segformer-onnx/model.onnx`: 수계 세그멘테이션 모델
- `models/segformer-onnx/preprocessor-config.json`: 세그멘테이션 모델 설정 파일
- `models/yolo11n.onnx`: 사람 디텍션 모델

### 실행

```bash
python run.py
```

서버는 기본적으로 `http://localhost:8000`에서 실행

### 빌드
```bash
python build_exe.py
```
기본적으로 dist 폴더에 결과물 생성

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
