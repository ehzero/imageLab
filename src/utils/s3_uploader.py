import os
import boto3
from datetime import datetime
from uuid import uuid4


def upload_to_s3(file_path: str, bucket_name: str = None) -> str:
    """
    S3에 이미지를 업로드하고 URL을 반환합니다.

    Parameters:
    - file_path: 업로드할 파일의 로컬 경로
    - bucket_name: S3 버킷 이름 (기본값: 환경변수 BUCKET_NAME)

    Returns:
    - S3 객체 URL
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

    s3_client.upload_file(file_path, bucket_name, s3_key)

    region = os.environ.get('AWS_REGION', 'ap-northeast-2')
    s3_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

    return s3_url
