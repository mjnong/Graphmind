from pathlib import Path
from typing import BinaryIO, Optional

import boto3

from .base import BaseStore

class S3Store(BaseStore):
    """
    Wraps any S3-compatible service.
    Requires environment variables or explicit kwargs
    for credentials & region (AWS works out of the box; MinIO needs endpoint_url).
    """

    def __init__(
        self,
        bucket: str,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        public: bool = False,
        prefix: str = "",
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        extra_cfg = {"region_name": region} if region else {}
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url, **extra_cfg)
        self.public = public  # if True: return raw https URL instead of presigned

    # ---------- helpers ---------- #
    def _key(self, object_key: str) -> str:
        return f"{self.prefix}/{object_key}" if self.prefix else object_key

    # ---------- API ---------- #
    def put(self, file: BinaryIO, object_key: str) -> None:
        key = self._key(object_key)
        self.s3.upload_fileobj(file, self.bucket, key)

    def get(self, object_key: str, dest: Path) -> Path:
        key = self._key(object_key)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(dest))
        return dest

    def delete(self, object_key: str) -> None:
        key = self._key(object_key)
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def url(self, object_key: str, expires: int = 3600) -> str:
        key = self._key(object_key)
        if self.public:
            # Works if bucket policy allows public read
            endpoint = self.s3.meta.endpoint_url or f"https://{self.bucket}.s3.amazonaws.com"
            return f"{endpoint.rstrip('/')}/{key}"
        # Pre-signed, time-limited URL
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )