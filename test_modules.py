import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.core.background_remover import load_model as load_bg_model, remove_background_image
from src.core.upscaler import load_model as load_upscale_model, upscale_image
import cv2

print("=" * 60)
print("이미지 처리 통합 테스트")
print("=" * 60)

input_image = "test/test.webp"

if not os.path.exists(input_image):
    print(f"\n✗ 테스트 이미지를 찾을 수 없습니다: {input_image}")
    sys.exit(1)

print("\n[1/2] 모델 로딩 중...")
try:
    bg_model = load_bg_model('u2net')
    print("  ✓ 배경 제거 모델 로드 완료")

    upscale_model = load_upscale_model()
    print("  ✓ 업스케일 모델 로드 완료")
except Exception as e:
    print(f"\n✗ 모델 로딩 실패: {e}")
    sys.exit(1)

print("\n[2/2] 통합 테스트 (업스케일 4x + 배경 제거)...")
print("  순서: 업스케일링 → 배경 제거")
try:
    output_both = "test/test_output_both_4x.png"

    print("  - 이미지 로딩 중...")
    image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    input_size = image.shape[:2]
    print(f"    입력 크기: {input_size}")

    print("  - 업스케일링 중 (4x)...")
    upscaled = upscale_image(image, upscale_model, scale=4)
    upscaled_size = upscaled.shape[:2]
    print(f"    업스케일 크기: {upscaled_size}")

    print("  - 임시 파일 저장 중...")
    temp_upscaled = 'test/temp_upscaled.png'
    cv2.imwrite(temp_upscaled, upscaled)
    del image, upscaled

    print("  - 배경 제거 중...")
    result = remove_background_image(temp_upscaled, bg_model)
    result.save(output_both)

    os.remove(temp_upscaled)

    print(f"\n  ✓ 통합 처리 성공!")
    print(f"  - 출력 파일: {output_both}")
    print(f"  - 최종 크기: {result.size}")
    print(f"  - 최종 모드: {result.mode}")
except Exception as e:
    print(f"\n  ✗ 통합 처리 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("테스트 완료!")
print("=" * 60)
