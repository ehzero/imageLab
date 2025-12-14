# PyTorch 라이브러리 불러오기
import torch  # PyTorch 메인 라이브러리
import torch.nn as nn  # 신경망 모듈
import torch.nn.functional as F  # 신경망 함수들


class REBNCONV(nn.Module):
    """합성곱 + 배치정규화 + ReLU 활성화 함수를 합친 기본 블록"""
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        # in_ch: 입력 채널 수, out_ch: 출력 채널 수, dirate: 확장 비율
        super(REBNCONV, self).__init__()  # 부모 클래스 초기화
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)  # 합성곱 레이어 생성
        self.bn_s1 = nn.BatchNorm2d(out_ch)  # 배치 정규화 레이어 생성
        self.relu_s1 = nn.ReLU(inplace=True)  # ReLU 활성화 함수 생성 (메모리 절약 모드)

    def forward(self, x):
        # 순전파 함수: 입력 데이터가 레이어를 통과하는 과정
        hx = x  # 입력 데이터를 hx 변수에 저장
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))  # 합성곱 -> 정규화 -> 활성화 순서로 적용
        return xout  # 결과 반환


def _upsample_like(src, tar):
    """src 이미지를 tar 이미지 크기에 맞춰 업샘플링(확대)하는 함수"""
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)  # 쌍선형 보간법으로 크기 조정
    return src  # 확대된 이미지 반환


class RSU7(nn.Module):
    """7개 레이어를 가진 Residual U-block (잔차 연결이 있는 U자 구조)"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        # in_ch: 입력 채널, mid_ch: 중간 채널, out_ch: 출력 채널
        super(RSU7, self).__init__()  # 부모 클래스 초기화
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # 입력 변환 블록

        # 인코더(다운샘플링) 부분 - 이미지 크기를 점점 줄이면서 특징 추출
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # 첫 번째 합성곱 블록
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 맥스풀링으로 크기 1/2로 축소

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)  # 두 번째 합성곱 블록
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 크기 1/2로 축소

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)  # 세 번째 합성곱 블록
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 크기 1/2로 축소

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)  # 네 번째 합성곱 블록
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 크기 1/2로 축소

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)  # 다섯 번째 합성곱 블록
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 크기 1/2로 축소

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)  # 여섯 번째 합성곱 블록
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 일곱 번째 합성곱 블록 (확장 비율 2)

        # 디코더(업샘플링) 부분 - 이미지 크기를 다시 키우면서 복원
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # 여섯 번째 디코더 블록 (채널 2배를 받음)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # 다섯 번째 디코더 블록
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # 네 번째 디코더 블록
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # 세 번째 디코더 블록
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # 두 번째 디코더 블록
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)  # 첫 번째 디코더 블록

    def forward(self, x):
        # 순전파: 데이터가 네트워크를 통과하는 과정
        hx = x  # 입력 데이터 저장
        hxin = self.rebnconvin(hx)  # 입력 변환

        # 인코더: 점점 작아지면서 특징 추출
        hx1 = self.rebnconv1(hxin)  # 첫 번째 레이어 통과
        hx = self.pool1(hx1)  # 크기 축소

        hx2 = self.rebnconv2(hx)  # 두 번째 레이어 통과
        hx = self.pool2(hx2)  # 크기 축소

        hx3 = self.rebnconv3(hx)  # 세 번째 레이어 통과
        hx = self.pool3(hx3)  # 크기 축소

        hx4 = self.rebnconv4(hx)  # 네 번째 레이어 통과
        hx = self.pool4(hx4)  # 크기 축소

        hx5 = self.rebnconv5(hx)  # 다섯 번째 레이어 통과
        hx = self.pool5(hx5)  # 크기 축소

        hx6 = self.rebnconv6(hx)  # 여섯 번째 레이어 통과
        hx7 = self.rebnconv7(hx6)  # 일곱 번째 레이어 통과 (가장 깊은 곳)

        # 디코더: 크기를 다시 키우면서 복원
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))  # hx7과 hx6을 채널 방향으로 합쳐서 디코더 통과
        hx6dup = _upsample_like(hx6d, hx5)  # hx5 크기에 맞춰 확대

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))  # 확대된 것과 인코더 결과 합쳐서 통과
        hx5dup = _upsample_like(hx5d, hx4)  # hx4 크기에 맞춰 확대

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))  # 합쳐서 통과
        hx4dup = _upsample_like(hx4d, hx3)  # hx3 크기에 맞춰 확대

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))  # 합쳐서 통과
        hx3dup = _upsample_like(hx3d, hx2)  # hx2 크기에 맞춰 확대

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))  # 합쳐서 통과
        hx2dup = _upsample_like(hx2d, hx1)  # hx1 크기에 맞춰 확대

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))  # 최종 디코더 통과

        return hx1d + hxin  # 디코더 결과와 입력을 더해서 반환 (잔차 연결)


class RSU6(nn.Module):
    """6개 레이어를 가진 Residual U-block"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # 입력 변환

        # 인코더 (5단계 다운샘플링)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 가장 깊은 레이어

        # 디코더 (업샘플링하면서 복원)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        # 인코더: 크기를 줄이면서 특징 추출
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)

        # 디코더: 크기를 키우면서 복원
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin  # 잔차 연결


class RSU5(nn.Module):
    """5개 레이어를 가진 Residual U-block"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        # 인코더 (4단계 다운샘플링)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        # 디코더
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        # 인코더
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)

        # 디코더
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):
    """4개 레이어를 가진 Residual U-block"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        # 인코더 (3단계 다운샘플링)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        # 디코더
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        # 인코더
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        # 디코더
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):
    """4개 레이어를 가진 Residual U-block (확장 합성곱 버전, 풀링 없음)"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        # 확장 합성곱으로 수용 영역 확대 (풀링 대신 사용)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)  # 확장 비율 2
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)  # 확장 비율 4
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)  # 확장 비율 8

        # 디코더 (확장 합성곱 사용)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        # 인코더 (크기는 유지하면서 넓은 영역 파악)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        # 디코더
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class U2NET(nn.Module):
    """U²-Net 전체 네트워크 구조 (배경 제거용 메인 모델)"""
    def __init__(self, in_ch=3, out_ch=1):
        # in_ch=3: RGB 이미지 입력, out_ch=1: 마스크 출력
        super(U2NET, self).__init__()

        # 인코더: 이미지를 점점 작게 만들면서 특징 추출
        self.stage1 = RSU7(in_ch, 32, 64)  # 1단계: 7레이어 RSU 블록
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 크기 1/2로 축소

        self.stage2 = RSU6(64, 32, 128)  # 2단계: 6레이어 RSU 블록
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)  # 3단계: 5레이어 RSU 블록
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)  # 4단계: 4레이어 RSU 블록
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)  # 5단계: 확장 합성곱 RSU 블록
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)  # 6단계: 가장 깊은 레이어

        # 디코더: 이미지를 다시 키우면서 복원
        self.stage5d = RSU4F(1024, 256, 512)  # 5단계 디코더
        self.stage4d = RSU4(1024, 128, 256)  # 4단계 디코더
        self.stage3d = RSU5(512, 64, 128)  # 3단계 디코더
        self.stage2d = RSU6(256, 32, 64)  # 2단계 디코더
        self.stage1d = RSU7(128, 16, 64)  # 1단계 디코더

        # 사이드 출력: 각 단계별로 마스크 생성 (최종 결과 개선용)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)  # 1단계 사이드 출력
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)  # 2단계 사이드 출력
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)  # 3단계 사이드 출력
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)  # 4단계 사이드 출력
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)  # 5단계 사이드 출력
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)  # 6단계 사이드 출력

        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)  # 모든 사이드 출력을 합쳐서 최종 마스크 생성

    def forward(self, x):
        # 입력 이미지가 네트워크를 통과하는 과정
        hx = x

        # 인코더: 점점 작아지면서 특징 추출
        hx1 = self.stage1(hx)  # 1단계
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)  # 2단계
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)  # 3단계
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)  # 4단계
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)  # 5단계
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)  # 6단계 (가장 깊은 곳)
        hx6up = _upsample_like(hx6, hx5)  # hx5 크기에 맞춰 확대

        # 디코더: 크기를 키우면서 복원
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # 인코더 결과와 합쳐서 디코더 통과
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # 사이드 출력: 각 단계의 결과를 마스크로 변환
        d1 = self.side1(hx1d)  # 1단계 마스크

        d2 = self.side2(hx2d)  # 2단계 마스크
        d2 = _upsample_like(d2, d1)  # d1 크기에 맞춤

        d3 = self.side3(hx3d)  # 3단계 마스크
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)  # 4단계 마스크
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)  # 5단계 마스크
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)  # 6단계 마스크
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))  # 모든 마스크를 합쳐서 최종 결과 생성

        # sigmoid로 0~1 사이 값으로 변환 (0=배경, 1=전경)
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


class U2NETP(nn.Module):
    """U²-Net의 경량 버전 (U²-Net-P, 더 빠르지만 정확도는 조금 낮음)"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        # 인코더 (채널 수를 줄여서 경량화)
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # 디코더
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        # 사이드 출력
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        # U2NET과 동일한 구조, 채널 수만 적음
        hx = x

        # 인코더
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # 디코더
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # 사이드 출력
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)
