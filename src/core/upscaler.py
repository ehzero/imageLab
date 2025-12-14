import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_model(model_path=None):
    """Real-ESRGAN 모델을 로드합니다."""
    if model_path is None:
        model_path = './weights/realesrgan/RealESRGAN_x4plus.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 가중치를 찾을 수 없습니다: {model_path}\n"
            f"다운로드: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth\n"
            f"저장 위치: weights/realesrgan/RealESRGAN_x4plus.pth"
        )

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    device = get_device()
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )

    return upsampler


def upscale_image(image, upsampler, scale=4):
    output, _ = upsampler.enhance(image, outscale=scale)
    return output

