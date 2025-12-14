from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import Literal
import os
import cv2
import urllib.request
from src.core.background_remover import load_model as load_bg_model, remove_background_image
from src.core.upscaler import load_model as load_upscale_model, upscale_image
from src.utils.s3_uploader import upload_to_s3

app = FastAPI()

bg_model = None
upscale_model = None


def init_models(need_bg: bool = False, need_upscale: bool = False):
    """필요한 모델만 로드합니다."""
    global bg_model, upscale_model

    if need_bg and bg_model is None:
        bg_model = load_bg_model('u2net')

    if need_upscale and upscale_model is None:
        upscale_model = load_upscale_model()


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
    # 필요한 모델만 로드
    need_bg = option in ['both', 'bg_remove']
    need_upscale = option in ['both', 'upscale']
    init_models(need_bg=need_bg, need_upscale=need_upscale)

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
