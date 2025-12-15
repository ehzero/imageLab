import os
import boto3
from datetime import datetime
from uuid import uuid4


def upload_to_s3(file_path: str, bucket_name: str = None, expiration: int = 3600) -> str:
    """
    S3에 이미지를 업로드하고 presigned URL을 반환합니다.

    Parameters:
    - file_path: 업로드할 파일의 로컬 경로
    - bucket_name: S3 버킷 이름 (기본값: 환경변수 BUCKET_NAME)
    - expiration: URL 유효 시간 (초 단위, 기본값: 3600초 = 1시간)

    Returns:
    - S3 presigned URL (임시 접근 가능한 URL)
    """
    if bucket_name is None:
        bucket_name = os.environ.get('BUCKET_NAME')
        if not bucket_name:
            raise ValueError("BUCKET_NAME 환경변수가 설정되지 않았습니다.")

    s3_client = boto3.client('s3')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid4())[:8]
    file_extension = os.path.splitext(file_path)[1]
    s3_key = f"processed/{timestamp}_{unique_id}{file_extension}"

    # S3에 파일 업로드 (ContentType 설정)
    content_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
    }
    content_type = content_type_map.get(file_extension.lower(), 'image/jpeg')
    
    s3_client.upload_file(
        file_path,
        bucket_name,
        s3_key,
        ExtraArgs={'ContentType': content_type}
    )

    # Presigned URL 생성 (임시 접근 가능한 URL)
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': s3_key},
        ExpiresIn=expiration
    )

    return presigned_url
