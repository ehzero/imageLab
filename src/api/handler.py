from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import Literal
from contextlib import asynccontextmanager
import os
import cv2
import urllib.request
from dotenv import load_dotenv
from src.core.background_remover import load_model as load_bg_model, remove_background_image
from src.core.upscaler import load_model as load_upscale_model, upscale_image
from src.utils.s3_uploader import upload_to_s3
from src.utils.model_downloader import ensure_model_weights

# .env 파일 로드
load_dotenv()

# 전역 변수로 모델 저장
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행되는 라이프사이클 함수"""
    # 서버 시작 시 실행
    print("=" * 50)
    print("서버 시작: 모델 로딩 중...")
    print("=" * 50)

    # 모델 가중치 확인 및 다운로드
    ensure_model_weights()

    # 모델 사전 로드 (캐싱)
    print("\n배경 제거 모델 로드 중...")
    models['bg_model'] = load_bg_model('u2net')
    print("✓ 배경 제거 모델 로드 완료")

    print("\n업스케일 모델 로드 중...")
    models['upscale_model'] = load_upscale_model()
    print("✓ 업스케일 모델 로드 완료")

    print("\n" + "=" * 50)
    print("서버 준비 완료! 요청을 받을 수 있습니다.")
    print("=" * 50 + "\n")

    yield  # 서버 실행

    # 서버 종료 시 실행
    print("서버 종료 중...")
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/process")
async def api_process(
    image_url: str = Form(...),
    option: Literal['both', 'upscale', 'bg_remove'] = Form('both'),
    scale: int = Form(4)
):
    """
    이미지 처리 API

    Parameters:
    - image_url: 입력 이미지 URL
    - option: 'both' (업스케일+배경제거), 'upscale' (업스케일만), 'bg_remove' (배경제거만)
    - scale: 업스케일 배율 (2, 3, 4)
    """
    # 캐시된 모델 사용
    bg_model = models.get('bg_model')
    upscale_model = models.get('upscale_model')

    # URL에서 이미지 다운로드
    temp_input = f"/tmp/input_{os.path.basename(image_url)}"
    urllib.request.urlretrieve(image_url, temp_input)

    temp_output = f"/tmp/output_{os.path.basename(image_url)}"

    try:
        if option == 'bg_remove':
            # 배경 제거만
            no_bg_image = remove_background_image(temp_input, bg_model)
            no_bg_image.save(temp_output)

        elif option == 'upscale':
            # 업스케일만
            image = cv2.imread(temp_input, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"이미지를 불러올 수 없습니다: {temp_input}")

            upscaled_image = upscale_image(image, upscale_model, scale)
            cv2.imwrite(temp_output, upscaled_image)

        elif option == 'both':
            # 업스케일 + 배경 제거
            # 1. 업스케일링
            image = cv2.imread(temp_input, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"이미지를 불러올 수 없습니다: {temp_input}")

            upscaled_image = upscale_image(image, upscale_model, scale)

            # 2. 임시 저장
            temp_upscaled = '/tmp/temp_upscaled.png'
            cv2.imwrite(temp_upscaled, upscaled_image)

            # 3. 배경 제거
            no_bg_image = remove_background_image(temp_upscaled, bg_model)
            no_bg_image.save(temp_output)

            # 4. 임시 파일 삭제
            os.remove(temp_upscaled)

        # S3에 업로드
        s3_url = upload_to_s3(temp_output)

        # 임시 파일 정리
        os.remove(temp_input)
        os.remove(temp_output)

        return JSONResponse(content={"url": s3_url})

    except Exception as e:
        # 오류 발생 시 임시 파일 정리
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise
