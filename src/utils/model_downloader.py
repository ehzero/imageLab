import os
import urllib.request
from pathlib import Path


def download_file(url: str, dest_path: str) -> None:
    """
    URL에서 파일을 다운로드합니다.

    Parameters:
    - url: 다운로드할 파일의 URL
    - dest_path: 저장할 경로
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"다운로드 중: {url}")
    print(f"저장 위치: {dest_path}")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\r진행률: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end='')

    urllib.request.urlretrieve(url, dest_path, show_progress)
    print("\n다운로드 완료!")


def ensure_model_weights() -> None:
    """
    모델 가중치 파일이 없으면 자동으로 다운로드합니다.
    """
    weights_dir = Path("weights")

    # U2Net 모델 가중치
    u2net_path = weights_dir / "u2net" / "u2net.pth"
    if not u2net_path.exists():
        print("U2Net 모델 가중치를 찾을 수 없습니다. 다운로드를 시작합니다...")
        download_file(
            "https://drive.google.com/uc?export=download&id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
            str(u2net_path)
        )
    else:
        print(f"U2Net 모델 가중치 확인: {u2net_path}")

    # RealESRGAN 모델 가중치
    realesrgan_path = weights_dir / "realesrgan" / "RealESRGAN_x4plus.pth"
    if not realesrgan_path.exists():
        print("RealESRGAN 모델 가중치를 찾을 수 없습니다. 다운로드를 시작합니다...")
        download_file(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            str(realesrgan_path)
        )
    else:
        print(f"RealESRGAN 모델 가중치 확인: {realesrgan_path}")
