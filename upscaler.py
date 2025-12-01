# 필요한 라이브러리 불러오기
import os  # 파일 경로 관련 기능
import cv2  # OpenCV 이미지 처리
from realesrgan import RealESRGANer  # Real-ESRGAN 업스케일러
from basicsr.archs.rrdbnet_arch import RRDBNet  # RRDBNet 모델 아키텍처


def load_model(model_path=None):
    """Real-ESRGAN 모델을 로드합니다."""
    # x4plus 모델은 2배, 3배, 4배 모두 지원 (outscale 파라미터로 조정)
    if model_path is None:  # 경로 지정 안 됐으면
        model_path = './saved_models/realesrgan/RealESRGAN_x4plus.pth'  # 기본 경로 설정

    if not os.path.exists(model_path):  # 모델 파일이 없으면
        raise FileNotFoundError(  # 에러 발생
            f"모델 가중치를 찾을 수 없습니다: {model_path}\n"
            f"다운로드: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth\n"
            f"저장 위치: saved_models/realesrgan/RealESRGAN_x4plus.pth"
        )

    # 모델 생성 (x4plus 모델은 항상 scale=4로 고정)
    # scale 파라미터는 모델 구조를 결정하므로, x4plus 모델 파일과 일치해야 함
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # 업스케일러 생성 (netscale=4, outscale은 upscale_image에서 지정)
    # RealESRGANer의 scale은 모델의 네이티브 스케일을 의미하며, x4plus는 항상 4
    upsampler = RealESRGANer(
        scale=4,  # 모델의 네이티브 스케일 (x4plus는 항상 4)
        model_path=model_path,
        model=model,
        tile=0,  # 타일링 없음 (작은 이미지용)
        tile_pad=10,
        pre_pad=0,
        half=False,  # FP32 (CPU 필수)
        gpu_id=-1  # CPU 강제 사용 (MPS 미지원)
    )

    return upsampler  # 업스케일러 반환


def upscale_image(image, upsampler, scale=4):
    """이미지를 업스케일링합니다."""
    h, w = image.shape[:2]  # 원본 이미지 크기 저장
    print(f"입력 이미지 크기: {w}x{h}")  # 진행 상황 출력

    # 업스케일링 실행
    print(f"{scale}x 업스케일링 시작...")  # 진행 상황 출력
    output, _ = upsampler.enhance(image, outscale=scale)  # 업스케일링 수행

    out_h, out_w = output.shape[:2]  # 출력 이미지 크기 저장
    print(f"출력 이미지 크기: {out_w}x{out_h}")  # 진행 상황 출력

    return output  # 업스케일된 이미지 반환

