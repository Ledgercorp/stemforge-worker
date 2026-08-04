from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import requests


WORKSPACE = Path(os.environ.get("STEMFORGE_WORKSPACE", "/runpod-volume/stemforge"))
WORKSPACE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "file"


def _s3_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed; S3 storage is unavailable.") from exc

    bucket = os.environ.get("STEMFORGE_S3_BUCKET")
    if not bucket:
        raise RuntimeError("STEMFORGE_S3_BUCKET is not configured.")

    endpoint = os.environ.get("STEMFORGE_S3_ENDPOINT_URL") or None
    region = os.environ.get("STEMFORGE_S3_REGION", "auto")
    return (
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=os.environ.get("STEMFORGE_S3_ACCESS_KEY_ID")
            or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("STEMFORGE_S3_SECRET_ACCESS_KEY")
            or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        ),
        bucket,
    )


def storage_status() -> dict:
    configured = bool(os.environ.get("STEMFORGE_S3_BUCKET"))
    return {
        "mode": "s3_signed_urls" if configured else "runpod_volume",
        "signed_transfer_ready": configured,
        "bucket_configured": configured,
        "output_root": str(OUTPUT_DIR),
        "note": (
            "Set STEMFORGE_S3_BUCKET and S3-compatible credentials to activate private expiring upload and download links."
            if not configured
            else "Private expiring transfer links are active."
        ),
    }


def create_upload_job(payload: dict) -> dict:
    filename = _slug(str(payload.get("filename") or "upload.bin"))
    content_type = str(payload.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream")
    ttl_seconds = max(60, min(int(payload.get("ttl_seconds", 3600)), 86400))
    prefix = _slug(str(payload.get("artist") or "default"))
    nonce = hashlib.sha256(os.urandom(32)).hexdigest()[:20]
    key = f"incoming/{prefix}/{nonce}/{filename}"
    client, bucket = _s3_client()
    url = client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=ttl_seconds,
    )
    return {"status": "completed", "storage_key": key, "upload_url": url, "method": "PUT", "headers": {"Content-Type": content_type}, "expires_in_seconds": ttl_seconds}


def create_download_job(payload: dict) -> dict:
    key = str(payload.get("storage_key") or "").strip()
    if not key:
        return {"status": "rejected", "reason": "storage_key is required."}
    ttl_seconds = max(60, min(int(payload.get("ttl_seconds", 3600)), 86400))
    client, bucket = _s3_client()
    url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl_seconds)
    return {"status": "completed", "storage_key": key, "download_url": url, "expires_in_seconds": ttl_seconds}


def delete_storage_objects(payload: dict) -> dict:
    keys = payload.get("storage_keys") or []
    if isinstance(keys, str):
        keys = [keys]
    keys = [str(key).strip() for key in keys if str(key).strip()]
    if not keys:
        return {"status": "rejected", "reason": "storage_keys is required."}
    client, bucket = _s3_client()
    client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True})
    return {"status": "completed", "deleted": keys}


def _download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=(20, 3600), stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
    return destination


def _download_storage_key(key: str, destination: Path) -> Path:
    client, bucket = _s3_client()
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return destination


def materialize_input(payload: dict, field: str, destination: Path, *, required: bool = True) -> Path | None:
    spec = payload.get(field)
    storage_key = payload.get(f"{field}_storage_key")
    url = payload.get(f"{field}_url")
    encoded = payload.get(f"{field}_base64")
    existing_path = payload.get(f"{field}_path")

    if isinstance(spec, dict):
        storage_key = storage_key or spec.get("storage_key")
        url = url or spec.get("url")
        encoded = encoded or spec.get("base64")
        existing_path = existing_path or spec.get("path")
    elif isinstance(spec, str) and spec.startswith(("https://", "http://")):
        url = url or spec

    if storage_key:
        return _download_storage_key(str(storage_key), destination)
    if url:
        return _download_url(str(url), destination)
    if encoded:
        try:
            raw = base64.b64decode(str(encoded), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 supplied for {field}.") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        return destination
    if existing_path:
        source = Path(str(existing_path)).resolve()
        root = WORKSPACE.resolve()
        if root != source and root not in source.parents:
            raise ValueError(f"{field}_path must be inside {root}.")
        if not source.exists():
            raise FileNotFoundError(str(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    if required:
        raise ValueError(f"{field}_url, {field}_storage_key, {field}_base64, or {field}_path is required.")
    return None


def publish_file(path: Path, *, artist: str = "default", category: str = "outputs", ttl_seconds: int = 86400) -> dict:
    path = Path(path).resolve()
    prefix = _slug(artist)
    category_slug = _slug(category)
    nonce = hashlib.sha256(f"{path}-{path.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    filename = _slug(path.name)

    if not os.environ.get("STEMFORGE_S3_BUCKET"):
        destination_dir = OUTPUT_DIR / category_slug / prefix / nonce
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / filename
        if path != destination:
            shutil.copy2(path, destination)
        return {
            "filename": destination.name,
            "local_path": str(destination),
            "volume_path": str(destination),
            "size_bytes": destination.stat().st_size,
            "download_url": None,
            "storage_mode": "runpod_volume",
            "storage_note": "Persisted on the RunPod network volume; use volume_file_chunk or configure S3 for signed downloads.",
        }

    client, bucket = _s3_client()
    key = f"{category_slug}/{prefix}/{nonce}/{filename}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": content_type})
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=max(60, min(int(ttl_seconds), 604800)),
    )
    return {
        "filename": path.name,
        "local_path": str(path),
        "size_bytes": path.stat().st_size,
        "storage_key": key,
        "download_url": url,
        "expires_in_seconds": max(60, min(int(ttl_seconds), 604800)),
        "storage_mode": "s3_signed_urls",
    }


def publish_files(paths: Iterable[Path], *, artist: str = "default", category: str = "outputs", ttl_seconds: int = 86400) -> list[dict]:
    return [publish_file(Path(path), artist=artist, category=category, ttl_seconds=ttl_seconds) for path in paths]
