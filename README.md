# 이미지 배경 제거 및 업스케일링 도구

U²-Net과 Real-ESRGAN을 사용한 이미지 배경 제거 및 업스케일링 통합 도구입니다.

## 특징

- **배경 제거**: U²-Net 모델을 사용한 고품질 배경 제거
- **이미지 업스케일링**: Real-ESRGAN을 사용한 AI 기반 업스케일링 (2배, 3배, 4배 지원)
- **통합 처리**: 배경 제거와 업스케일링을 한 번에 수행
- **선택적 처리**: 배경 제거만 또는 업스케일링만 수행 가능
- **CPU/GPU 자동 감지**: CUDA 사용 가능 시 자동으로 GPU 활용
- **투명 배경 지원**: PNG 알파 채널 보존

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/ehzero/BGRemoveUpscale.git
cd BGRemoveUpscale
```

### 2. 가상 환경 설정 (권장)

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 사전 학습된 모델 다운로드

**배경 제거 모델 (필수 중 하나):**

- **u2net.pth** (권장, 176.3 MB): [다운로드](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing)
- **u2netp.pth** (경량, 4.7 MB): [다운로드](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing)

저장 위치: `saved_models/u2net/u2net.pth` 또는 `saved_models/u2net/u2netp.pth`

**업스케일링 모델 (필수):**

- **RealESRGAN_x4plus.pth** (64 MB): [다운로드](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

저장 위치: `saved_models/realesrgan/RealESRGAN_x4plus.pth`

## 사용법

### 기본 사용 (배경 제거 + 업스케일링)

```bash
# 기본: 배경 제거 후 4배 업스케일링
python main.py --input image.jpg

# 2배 업스케일링
python main.py --input image.jpg --scale 2

# 출력 파일명 지정
python main.py --input image.jpg --output result.png --scale 4
```

### 배경 제거만 수행

```bash
# 배경 제거만 수행 (업스케일링 제외)
python main.py --input image.jpg --bg-only

# 경량 모델 사용
python main.py --input image.jpg --bg-only --bg-model u2netp
```

### 업스케일링만 수행

```bash
# 업스케일링만 수행 (배경 제거 제외)
python main.py --input image.jpg --upscale-only --scale 2

# 4배 업스케일링
python main.py --input image.jpg --upscale-only --scale 4
```

### 명령줄 인자

**필수:**

- `--input, -i`: 입력 이미지 파일 경로

**선택:**

- `--output, -o`: 출력 이미지 파일 경로 (기본값: 자동 생성)
- `--bg-only`: 배경 제거만 수행 (업스케일링 제외)
- `--upscale-only`: 업스케일링만 수행 (배경 제거 제외)
- `--bg-model, -b`: 배경 제거 모델 (`u2net` 또는 `u2netp`, 기본값: `u2net`)
- `--bg-model-path`: 배경 제거 모델 가중치 파일 경로 (선택사항)
- `--scale, -s`: 업스케일 배율 (`2`, `3`, `4`, 기본값: `4`)
- `--upscale-model-path`: 업스케일 모델 가중치 파일 경로 (선택사항)

### 출력 파일명 규칙

- 배경 제거만: `{입력파일명}_nobg.png`
- 업스케일링만: `{입력파일명}_upscaled_{배율}x.png`
- 둘 다 수행: `{입력파일명}_processed_{배율}x.png`

## 프로젝트 구조

```
BGRemoveUpscale/
├── model/
│   └── u2net.py          # U²-Net 모델 아키텍처
├── saved_models/
│   ├── u2net/            # 배경 제거 모델 가중치
│   │   ├── u2net.pth
│   │   └── u2netp.pth
│   └── realesrgan/        # 업스케일링 모델 가중치
│       └── RealESRGAN_x4plus.pth
├── background_remover.py  # 배경 제거 모듈
├── upscaler.py            # 업스케일링 모듈
├── main.py                # 통합 메인 스크립트
├── requirements.txt       # 의존성
└── README.md             # 사용 설명서
```

## Python 코드에서 사용하기

```python
from background_remover import load_model as load_bg_model, remove_background_image
from upscaler import load_model as load_upscale_model, upscale_image
from main import pil_to_cv2

# 모델 로드
bg_model = load_bg_model('u2net')
upscale_model = load_upscale_model()

# 배경 제거
no_bg_image = remove_background_image('input.jpg', bg_model)

# 업스케일링
cv2_image = pil_to_cv2(no_bg_image)
upscaled_image = upscale_image(cv2_image, upscale_model, scale=4)
```

## 시스템 요구사항

- Python 3.7+
- PyTorch 1.7.0+
- **CPU**: 4GB+ RAM
- **GPU (CUDA)**: 6GB+ VRAM (선택사항, 더 빠른 처리)

## 지원 이미지 형식

**입력:** JPG, PNG, BMP, TIFF, WebP 등 PIL/OpenCV가 지원하는 모든 형식

**출력:** PNG (투명 배경 지원)

## 성능

- **배경 제거**: CPU에서 약 1-2초 (360x360 이미지 기준)
- **업스케일링**: CPU에서 약 25초 (360x360 → 1440x1440)
- **통합 처리**: CPU에서 약 26-27초 (360x360 → 1440x1440)

## 참고

- U²-Net: [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
- Real-ESRGAN: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## 라이선스

이 프로젝트는 U²-Net과 Real-ESRGAN의 공식 구현을 기반으로 합니다.
