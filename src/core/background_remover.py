import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from src.models.u2net_arch.u2net import U2NET, U2NETP


def load_model(model_name='u2net', model_path=None):
    """U²-Net 모델을 로드합니다."""
    if model_name == 'u2net':
        model = U2NET(3, 1)
        if model_path is None:
            model_path = './weights/u2net/u2net.pth'
    elif model_name == 'u2netp':
        model = U2NETP(3, 1)
        if model_path is None:
            model_path = './weights/u2net/u2netp.pth'
    else:
        raise ValueError(f"알 수 없는 모델 이름: {model_name}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 가중치를 찾을 수 없습니다: {model_path}\n"
            f"다운로드: https://github.com/xuebinqin/U-2-Net\n"
            f"- u2net.pth (176.3 MB): 전체 모델\n"
            f"- u2netp.pth (4.7 MB): 경량 모델"
        )

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    model.eval()
    return model


def normalize_image(image):
    """이미지를 모델 입력 형식으로 정규화합니다."""
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


def remove_background_image(image_path, model):
    """이미지의 배경을 제거하고 PIL Image 객체를 반환합니다."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    input_tensor = normalize_image(image)
    input_tensor = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = model(input_tensor)

    pred = d1[:, 0, :, :].squeeze().cpu().data.numpy()
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)
    mask_np = np.array(mask)
    image_np = np.array(image)
    result = np.dstack((image_np, mask_np))
    result_image = Image.fromarray(result.astype(np.uint8), mode='RGBA')

    return result_image

