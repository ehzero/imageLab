# 필요한 라이브러리 불러오기
import os  # 파일 경로 관련 기능
import argparse  # 명령줄 인자 파싱
import traceback  # 상세 오류 추적 모듈
import cv2  # OpenCV 이미지 처리
from background_remover import load_model as load_bg_removal_model, remove_background_image, pil_to_cv2  # 배경 제거 모듈
from upscaler import load_model as load_upscale_model, upscale_image  # 업스케일 모듈


def save_image_cv2(image, output_path, input_path, suffix='processed'):
    """OpenCV 형식 이미지를 저장합니다."""
    # 출력 경로 생성
    if output_path is None:  # 출력 경로가 지정 안 됐으면
        base_name, ext = os.path.splitext(input_path)  # 입력 파일명에서 확장자 분리
        output_path = f"{base_name}_{suffix}.png"  # 자동으로 파일명 생성

    # 결과 저장
    cv2.imwrite(output_path, image)  # 이미지 저장
    print(f"처리 완료: {output_path}")  # 완료 메시지 출력

    return output_path  # 저장된 파일 경로 반환


def save_image_pil(image, output_path, input_path, suffix='nobg'):
    """PIL Image를 저장합니다."""
    # 출력 경로 생성
    if output_path is None:  # 출력 경로가 지정 안 됐으면
        base_name, ext = os.path.splitext(input_path)  # 입력 파일명에서 확장자 분리
        output_path = f"{base_name}_{suffix}.png"  # 자동으로 파일명 생성

    # 결과 저장
    image.save(output_path)  # 이미지 저장
    print(f"처리 완료: {output_path}")  # 완료 메시지 출력

    return output_path  # 저장된 파일 경로 반환


def process_bg_only(image_path, bg_model, output_path=None):
    """배경 제거만 수행합니다."""
    print(f"배경 제거 중: {image_path}")  # 진행 상황 출력
    no_bg_image = remove_background_image(image_path, bg_model)  # 배경 제거 수행
    print("배경 제거 완료!")  # 완료 메시지 출력

    # 결과 저장
    output_path = save_image_pil(no_bg_image, output_path, image_path, 'nobg')  # 결과 저장

    return output_path  # 저장된 파일 경로 반환


def process_upscale_only(image_path, upscale_model, scale=4, output_path=None):
    """업스케일링만 수행합니다."""
    # 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 이미지 읽기
    if image is None:  # 이미지 로드 실패 시
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 업스케일링 수행
    upscaled_image = upscale_image(image, upscale_model, scale)  # 업스케일링 수행

    # 결과 저장
    suffix = f'upscaled_{scale}x'  # 파일명 접미사
    output_path = save_image_cv2(upscaled_image, output_path, image_path, suffix)  # 결과 저장

    return output_path  # 저장된 파일 경로 반환


def process_both(image_path, bg_model, upscale_model, scale=4, output_path=None):
    """이미지의 배경을 제거하고 업스케일링합니다."""
    # 배경 제거 수행
    print(f"배경 제거 중: {image_path}")  # 진행 상황 출력
    no_bg_image = remove_background_image(image_path, bg_model)  # 배경 제거 수행
    print("배경 제거 완료!")  # 완료 메시지 출력

    # PIL Image를 OpenCV 형식으로 변환
    cv2_image = pil_to_cv2(no_bg_image)  # 형식 변환

    # 업스케일링 수행
    upscaled_image = upscale_image(cv2_image, upscale_model, scale)  # 업스케일링 수행

    # 결과 저장
    suffix = f'processed_{scale}x'  # 파일명 접미사
    output_path = save_image_cv2(upscaled_image, output_path, image_path, suffix)  # 결과 저장

    return output_path  # 저장된 파일 경로 반환


def main():
    parser = argparse.ArgumentParser(description='이미지 배경 제거 및 업스케일링 통합 처리')
    parser.add_argument('--input', '-i', required=True, help='입력 이미지 파일 경로')
    parser.add_argument('--output', '-o', help='출력 이미지 파일 경로 (기본값: 자동 생성)')
    parser.add_argument('--bg-only', action='store_true', help='배경 제거만 수행 (업스케일링 제외)')
    parser.add_argument('--upscale-only', action='store_true', help='업스케일링만 수행 (배경 제거 제외)')
    parser.add_argument('--bg-model', '-b', default='u2net', choices=['u2net', 'u2netp'], help='배경 제거 모델: u2net (전체) 또는 u2netp (경량)')
    parser.add_argument('--bg-model-path', help='배경 제거 모델 가중치 파일 경로 (선택사항)')
    parser.add_argument('--scale', '-s', type=int, default=4, choices=[2, 3, 4], help='업스케일 배율: 2, 3, 4 (기본값: 4)')
    parser.add_argument('--upscale-model-path', help='업스케일 모델 가중치 파일 경로 (선택사항)')
    args = parser.parse_args()

    # 옵션 충돌 확인
    if args.bg_only and args.upscale_only:  # 둘 다 True면
        print("오류: --bg-only와 --upscale-only는 동시에 사용할 수 없습니다.")  # 오류 메시지
        return  # 프로그램 종료

    # 입력 파일 존재 확인
    if not os.path.exists(args.input):  # 입력 파일이 없으면
        print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input}")  # 오류 메시지
        return  # 프로그램 종료

    # 배경 제거 모델 불러오기 (bg-only 또는 둘 다 수행할 때만)
    bg_model = None
    if not args.upscale_only:  # 업스케일링만이 아니면
        print(f"{args.bg_model} 모델 불러오는 중...")  # 진행 상황 출력
        try:  # 예외 처리 시작
            bg_model = load_bg_removal_model(args.bg_model, args.bg_model_path)  # 모델 로드 시도
            print("배경 제거 모델 로드 완료!")  # 성공 메시지
        except FileNotFoundError as e:  # 파일을 찾을 수 없는 경우
            print(f"오류: {e}")  # 오류 메시지 출력
            return  # 프로그램 종료
        except Exception as e:  # 기타 예외 발생 시
            print(f"배경 제거 모델 로드 오류: {e}")  # 오류 메시지 출력
            return  # 프로그램 종료

    # 업스케일 모델 불러오기 (upscale-only 또는 둘 다 수행할 때만)
    upscale_model = None
    if not args.bg_only:  # 배경 제거만이 아니면
        print("RealESRGAN_x4plus 모델 불러오는 중...")  # 진행 상황 출력
        try:  # 예외 처리 시작
            upscale_model = load_upscale_model(args.upscale_model_path)  # 모델 로드 시도
            print("업스케일 모델 로드 완료!")  # 성공 메시지
        except FileNotFoundError as e:  # 파일을 찾을 수 없는 경우
            print(f"오류: {e}")  # 오류 메시지 출력
            return  # 프로그램 종료
        except Exception as e:  # 기타 예외 발생 시
            print(f"업스케일 모델 로드 오류: {e}")  # 오류 메시지 출력
            return  # 프로그램 종료

    # 이미지 처리 실행
    print(f"이미지 처리 중: {args.input}")  # 진행 상황 출력
    try:  # 예외 처리 시작
        if args.bg_only:  # 배경 제거만 수행
            process_bg_only(args.input, bg_model, args.output)  # 배경 제거만 수행
        elif args.upscale_only:  # 업스케일링만 수행
            process_upscale_only(args.input, upscale_model, args.scale, args.output)  # 업스케일링만 수행
        else:  # 둘 다 수행 (기본 동작)
            process_both(args.input, bg_model, upscale_model, args.scale, args.output)  # 둘 다 수행
    except Exception as e:  # 예외 발생 시
        print(f"이미지 처리 오류: {e}")  # 오류 메시지 출력
        traceback.print_exc()  # 전체 오류 스택 출력


if __name__ == '__main__':  # 이 파일이 직접 실행될 때만
    main()  # 메인 함수 호출
