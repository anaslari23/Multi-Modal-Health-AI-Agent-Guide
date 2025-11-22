from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config


class S3Client:
    """Thin wrapper around boto3 for uploads and presigned URLs.

    Configuration is taken from environment variables:
      - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or IAM role)
      - AWS_DEFAULT_REGION or AWS_REGION
      - S3_BUCKET_NAME (required)
      - S3_ENDPOINT_URL (optional, for custom endpoints / MinIO)
    """

    def __init__(self, bucket_name: str, region: Optional[str] = None, endpoint_url: Optional[str] = None) -> None:
        session = boto3.session.Session()
        self.bucket_name = bucket_name
        self.s3 = session.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url,
            config=Config(s3={"addressing_style": "virtual"}),
        )

    @classmethod
    def from_env(cls) -> "S3Client":
        import os

        bucket = os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise RuntimeError("S3_BUCKET_NAME environment variable is required for S3 uploads")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        endpoint = os.getenv("S3_ENDPOINT_URL")
        return cls(bucket_name=bucket, region=region, endpoint_url=endpoint)

    def upload_file(self, path: Path, key: str) -> str:
        """Upload a local file to S3 under the given key.

        Returns the key that was used.
        """

        self.s3.upload_file(str(path), self.bucket_name, key)
        return key

    def upload_bytes(self, data: bytes, key: str, content_type: Optional[str] = None) -> str:
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=data, **extra_args)
        return key

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": key},
            ExpiresIn=expires_in,
        )


_s3_client: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    global _s3_client
    if _s3_client is None:
        _s3_client = S3Client.from_env()
    return _s3_client
