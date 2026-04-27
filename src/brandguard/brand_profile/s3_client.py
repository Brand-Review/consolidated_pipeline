"""
S3 storage for raw uploads and processed plaintext JSON.
Reuses br-be's bucket (env: S3_BUCKET_NAME, AWS_REGION, AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY). All network calls are lazy — no boto3 client is created
until an upload/download is actually requested.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class S3Client:
    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        prefix: str = "brand-onboarding",
    ):
        self.bucket = bucket or os.environ.get("S3_BUCKET_NAME")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.prefix = prefix.strip("/")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError("boto3 is required: pip install boto3") from e
            kwargs: Dict[str, Any] = {"region_name": self.region}
            aki = os.environ.get("AWS_ACCESS_KEY_ID")
            sak = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if aki and sak:
                kwargs["aws_access_key_id"] = aki
                kwargs["aws_secret_access_key"] = sak
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _raw_key(self, brand_id: str, doc_id: str, filename: str) -> str:
        return f"{self.prefix}/{brand_id}/{doc_id}/raw/{filename}"

    def _processed_key(self, brand_id: str, doc_id: str) -> str:
        return f"{self.prefix}/{brand_id}/{doc_id}/processed/plaintext.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_raw(
        self, brand_id: str, doc_id: str, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> str:
        if not self.bucket:
            logger.warning("S3Client.upload_raw called without a bucket; skipping upload")
            return ""
        key = self._raw_key(brand_id, doc_id, filename)
        extra = {"ContentType": content_type} if content_type else {}
        self._get_client().put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        logger.info(f"Uploaded raw doc to s3://{self.bucket}/{key}")
        return key

    def upload_processed(self, brand_id: str, doc_id: str, payload: Dict[str, Any]) -> str:
        if not self.bucket:
            logger.warning("S3Client.upload_processed called without a bucket; skipping upload")
            return ""
        key = self._processed_key(brand_id, doc_id)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._get_client().put_object(
            Bucket=self.bucket, Key=key, Body=body, ContentType="application/json"
        )
        logger.info(f"Uploaded processed doc to s3://{self.bucket}/{key}")
        return key

    def download(self, key: str) -> bytes:
        if not self.bucket:
            raise RuntimeError("S3Client.download called without a bucket")
        resp = self._get_client().get_object(Bucket=self.bucket, Key=key)
        return resp["Body"].read()
