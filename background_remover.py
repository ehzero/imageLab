# 필요한 라이브러리 불러오기
import os  # 파일 경로 관련 기능
import numpy as np  # 수치 연산 (배열 처리)
import cv2  # OpenCV 이미지 처리
from PIL import Image  # 이미지 읽기/쓰기
import torch  # PyTorch 딥러닝 프레임워크
from torchvision import transforms  # 이미지 변환 도구
from model.u2net import U2NET, U2NETP  # U²-Net 모델 불러오기


def load_model(model_name='u2net', model_path=None):
    """U²-Net 모델을 로드합니다."""
    if model_name == 'u2net':  # 전체 모델 선택 시
        model = U2NET(3, 1)  # U²-Net 모델 생성 (입력 3채널, 출력 1채널)
        if model_path is None:  # 경로 지정 안 됐으면
            model_path = './saved_models/u2net/u2net.pth'  # 기본 경로 설정
    elif model_name == 'u2netp':  # 경량 모델 선택 시
        model = U2NETP(3, 1)  # U²-Net-P 모델 생성
        if model_path is None:
            model_path = './saved_models/u2net/u2netp.pth'  # 기본 경로 설정
    else:  # 잘못된 모델 이름
        raise ValueError(f"알 수 없는 모델 이름: {model_name}")

    if not os.path.exists(model_path):  # 모델 파일이 없으면
        raise FileNotFoundError(  # 에러 발생
            f"모델 가중치를 찾을 수 없습니다: {model_path}\n"
            f"다운로드: https://github.com/xuebinqin/U-2-Net\n"
            f"- u2net.pth (176.3 MB): 전체 모델\n"
            f"- u2netp.pth (4.7 MB): 경량 모델"
        )

    # 모델 가중치 불러오기
    if torch.cuda.is_available():  # GPU가 사용 가능하면
        model.load_state_dict(torch.load(model_path))  # GPU로 가중치 불러오기
        model.cuda()  # 모델을 GPU로 이동
    else:  # CPU만 사용 가능하면
        model.load_state_dict(torch.load(model_path, map_location='cpu'))  # CPU로 가중치 불러오기

    model.eval()  # 모델을 평가 모드로 설정 (학습 안 함)
    return model  # 모델 반환


def normalize_image(image):
    """이미지를 모델 입력 형식으로 정규화합니다."""
    transform = transforms.Compose([  # 여러 변환을 순서대로 적용
        transforms.Resize((320, 320)),  # 이미지 크기를 320x320으로 조정
        transforms.ToTensor(),  # PIL 이미지를 텐서로 변환 (0~1 범위로 정규화)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # RGB 평균값으로 정규화
                           std=[0.229, 0.224, 0.225])  # RGB 표준편차로 정규화
    ])
    return transform(image)  # 변환 적용 후 반환


def remove_background_image(image_path, model):
    """이미지의 배경을 제거하고 PIL Image 객체를 반환합니다."""
    # 이미지 불러오기
    image = Image.open(image_path).convert('RGB')  # 이미지를 RGB 형식으로 열기
    original_size = image.size  # 원본 이미지 크기 저장 (나중에 복원용)

    # 전처리
    input_tensor = normalize_image(image)  # 이미지 정규화
    input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가 [3, 320, 320] -> [1, 3, 320, 320]

    if torch.cuda.is_available():  # GPU 사용 가능하면
        input_tensor = input_tensor.cuda()  # 텐서를 GPU로 이동

    # 추론 (배경 제거 수행)
    with torch.no_grad():  # 기울기 계산 안 함 (학습 안 하므로)
        d1, d2, d3, d4, d5, d6, d7 = model(input_tensor)  # 모델 실행 (7개 출력)

    # 예측 결과 가져오기 (d1이 최종 출력)
    pred = d1[:, 0, :, :]  # 배치와 채널 차원에서 데이터 추출
    pred = pred.squeeze().cpu().data.numpy()  # GPU -> CPU -> NumPy 배열로 변환

    # 마스크를 원본 이미지 크기로 조정
    mask = Image.fromarray((pred * 255).astype(np.uint8)).resize(original_size, Image.BILINEAR)  # 0~1 값을 0~255로 변환 후 리사이즈
    mask_np = np.array(mask)  # PIL 이미지를 NumPy 배열로 변환

    # 마스크를 원본 이미지에 적용하여 투명 배경 생성
    image_np = np.array(image)  # 원본 이미지를 NumPy 배열로 변환

    # RGBA 이미지 생성 (RGB + Alpha 채널)
    result = np.dstack((image_np, mask_np))  # RGB 이미지에 마스크를 알파 채널로 추가
    result_image = Image.fromarray(result.astype(np.uint8), mode='RGBA')  # NumPy 배열을 RGBA 이미지로 변환

    return result_image  # 배경 제거된 이미지 반환


def pil_to_cv2(pil_image):
    """PIL Image를 OpenCV 형식으로 변환합니다."""
    if pil_image.mode == 'RGBA':  # RGBA 이미지인 경우
        # RGBA를 BGRA로 변환 (OpenCV는 BGR 순서 사용)
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)  # RGBA -> BGRA 변환
    else:  # RGB 이미지인 경우
        # RGB를 BGR로 변환
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환

    return cv2_image  # OpenCV 형식 이미지 반환

