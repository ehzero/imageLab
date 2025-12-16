### ImageLab

FastAPI 기반의 **이미지 업스케일(Real-ESRGAN)** 및 **배경 제거(U²-Net)** API 서버입니다. 처리된 결과 이미지는 **S3에 업로드**되고 presigned URL을 반환합니다.

### 기능

- **업스케일**: Real-ESRGAN (`src/core/upscaler.py`)
- **배경 제거**: U²-Net (`src/core/background_remover.py`)
- **S3 업로드**: 업로드 후 presigned URL 반환 (`src/utils/s3_uploader.py`)
- **단일 엔드포인트**: `POST /process` (`src/api/handler.py`)

### 요구사항

- Python 3.11
- `pip install -r requirements.txt`

### 환경변수

`env.example`을 참고해 `.env`를 생성하세요.

- **필수**
  - `BUCKET_NAME`
- **권장**
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION`

### 실행

```bash
uvicorn src.api.handler:app --reload
```

서버 시작 시 `weights/` 내 모델 파일을 확인하고(필요 시) 다운로드합니다.

### API

#### `POST /process` (multipart/form-data)

- **image_url**: 처리할 이미지 URL
- **option**: `both` | `upscale` | `bg_remove` (기본 `both`)
- **scale**: 업스케일 배율 (기본 `4`)

예시:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "image_url=https://example.com/sample.webp" \
  -F "option=both" \
  -F "scale=4"
```

응답:

```json
{ "url": "<presigned-url>" }
```

### 참고

- `bg_remove`, `both` 옵션은 **알파(투명) 채널**이 포함될 수 있어 출력은 **PNG로 저장**됩니다.
- 로컬 통합 테스트 스크립트: `test_modules.py`
