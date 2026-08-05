from __future__ import annotations

import base64
import binascii
import hashlib
import mimetypes
import os
import re
import shutil
import time
from pathlib import Path
from typing import Iterable

import requests


WORKSPACE = Path(os.environ.get("STEMFORGE_WORKSPACE", "/runpod-volume/stemforge"))
WORKSPACE.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class InputTransferError(RuntimeError):
    """Raised when an input cannot be fetched or fails integrity validation."""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "file"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            config=Config(signature_version="s3v4", retries={"max_attempts": 4}),
        ),
        bucket,
    )


def storage_status() -> dict:
    """Report storage readiness, including whether a client can be built.

    Checking only that STEMFORGE_S3_BUCKET is set reports success while the
    credentials are absent or incomplete, so an operator sees
    signed_transfer_ready and every later upload still fails with
    PartialCredentialsError. Construct the client here: it signs URLs locally
    and makes no network call, so this stays cheap while proving the
    configuration is usable rather than merely present.
    """
    configured = bool(os.environ.get("STEMFORGE_S3_BUCKET"))
    client_error: str | None = None
    if configured:
        try:
            # Constructing the client proves nothing: boto3 resolves
            # credentials lazily, so client() succeeds with none present and
            # only signing raises NoCredentialsError. Sign a throwaway URL
            # instead - still local, still no network call, but it exercises
            # the credential path the real upload will take.
            client, bucket = _s3_client()
            client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": "__readiness_probe__"},
                ExpiresIn=60,
            )
        except Exception as exc:  # boto3 absent, or credentials missing
            client_error = f"{type(exc).__name__}: {exc}"

    ready = configured and client_error is None
    if not configured:
        note = (
            "Set STEMFORGE_S3_BUCKET and S3-compatible credentials to activate "
            "private expiring upload and download links."
        )
    elif client_error:
        note = (
            "STEMFORGE_S3_BUCKET is set but no S3 client could be built, so "
            "uploads and downloads will fail. Check "
            "STEMFORGE_S3_ACCESS_KEY_ID, STEMFORGE_S3_SECRET_ACCESS_KEY and "
            "STEMFORGE_S3_ENDPOINT_URL."
        )
    else:
        note = "Private expiring transfer links are active."

    status = {
        "mode": "s3_signed_urls" if ready else "runpod_volume",
        "signed_transfer_ready": ready,
        "bucket_configured": configured,
        "client_ready": ready,
        "output_root": str(OUTPUT_DIR),
        "note": note,
        # Which settings actually reached the worker. Names and lengths only:
        # enough to find an empty box, a typo or a truncated paste without
        # ever putting a credential in a job result.
        "configuration": _configuration_report(),
    }
    if client_error:
        status["client_error"] = client_error
    return status


S3_SETTINGS = (
    "STEMFORGE_S3_BUCKET",
    "STEMFORGE_S3_ENDPOINT_URL",
    "STEMFORGE_S3_REGION",
    "STEMFORGE_S3_ACCESS_KEY_ID",
    "STEMFORGE_S3_SECRET_ACCESS_KEY",
)


def _configuration_report() -> dict:
    """Report presence and shape of each storage setting, never its value.

    Diagnosing a misconfigured bucket otherwise means guessing which of five
    variables is wrong. A name, a length, and for non-secret settings the
    value itself are enough to spot an empty box, a stray space or a
    truncated paste - and none of that discloses a credential.
    """
    secret = {"STEMFORGE_S3_ACCESS_KEY_ID", "STEMFORGE_S3_SECRET_ACCESS_KEY"}
    report: dict[str, dict] = {}
    for name in S3_SETTINGS:
        raw = os.environ.get(name)
        entry: dict = {"set": raw is not None, "empty": not (raw or "").strip()}
        if raw is not None:
            entry["length"] = len(raw)
            entry["has_surrounding_whitespace"] = raw != raw.strip()
            if name not in secret:
                entry["value"] = raw
        report[name] = entry

    # A credential landing under the AWS names instead is a common mix-up.
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if os.environ.get(name):
            report[name] = {"set": True, "empty": False, "length": len(os.environ[name])}
    return report


def create_upload_job(payload: dict) -> dict:
    filename = _slug(str(payload.get("filename") or "upload.bin"))
    content_type = str(
        payload.get("content_type")
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )
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
    return {
        "status": "completed",
        "storage_key": key,
        "upload_url": url,
        "method": "PUT",
        "headers": {"Content-Type": content_type},
        "expires_in_seconds": ttl_seconds,
    }


def create_download_job(payload: dict) -> dict:
    key = str(payload.get("storage_key") or "").strip()
    if not key:
        return {"status": "rejected", "reason": "storage_key is required."}
    ttl_seconds = max(60, min(int(payload.get("ttl_seconds", 3600)), 86400))
    client, bucket = _s3_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl_seconds,
    )
    return {
        "status": "completed",
        "storage_key": key,
        "download_url": url,
        "expires_in_seconds": ttl_seconds,
    }


def delete_storage_objects(payload: dict) -> dict:
    keys = payload.get("storage_keys") or []
    if isinstance(keys, str):
        keys = [keys]
    keys = [str(key).strip() for key in keys if str(key).strip()]
    if not keys:
        return {"status": "rejected", "reason": "storage_keys is required."}
    client, bucket = _s3_client()
    client.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": key} for key in keys], "Quiet": True},
    )
    return {"status": "completed", "deleted": keys}


def _validate_download(
    path: Path,
    *,
    content_type: str | None,
    expected_size_bytes: int | None,
    expected_sha256: str | None,
) -> None:
    size = path.stat().st_size
    if size <= 0:
        raise InputTransferError("Downloaded input is empty.")

    first_bytes = path.read_bytes()[:512].lstrip().lower()
    if (
        (content_type or "").lower().startswith("text/html")
        or first_bytes.startswith(b"<!doctype html")
        or first_bytes.startswith(b"<html")
    ):
        raise InputTransferError(
            "The input URL returned an HTML page instead of the requested file. "
            "The link may be private, expired, or require browser confirmation."
        )

    if expected_size_bytes is not None and size != int(expected_size_bytes):
        raise InputTransferError(
            f"Input size mismatch: expected {int(expected_size_bytes)} bytes, received {size}."
        )

    if expected_sha256:
        actual = _sha256(path)
        if actual.lower() != str(expected_sha256).lower():
            raise InputTransferError(
                f"Input SHA-256 mismatch: expected {expected_sha256}, received {actual}."
            )


def _download_url(
    url: str,
    destination: Path,
    *,
    expected_size_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with requests.get(
                url,
                timeout=(20, 3600),
                stream=True,
                allow_redirects=True,
                headers={"User-Agent": "StemForge/2.2.1"},
            ) as response:
                if response.status_code == 404:
                    raise InputTransferError(
                        "Input URL returned HTTP 404. The temporary file may have been deleted or the link is stale."
                    )
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                _validate_download(
                    temporary,
                    content_type=response.headers.get("Content-Type"),
                    expected_size_bytes=expected_size_bytes,
                    expected_sha256=expected_sha256,
                )
                temporary.replace(destination)
                return destination
        except InputTransferError:
            temporary.unlink(missing_ok=True)
            raise
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(float(2 ** (attempt - 1)))

    raise InputTransferError(
        f"Input download failed after 3 attempts: {type(last_error).__name__}: {last_error}"
    )


def _download_storage_key(key: str, destination: Path) -> Path:
    client, bucket = _s3_client()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        client.download_file(bucket, key, str(temporary))
        _validate_download(
            temporary,
            content_type=None,
            expected_size_bytes=None,
            expected_sha256=None,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def materialize_input(
    payload: dict,
    field: str,
    destination: Path,
    *,
    required: bool = True,
) -> Path | None:
    spec = payload.get(field)
    storage_key = payload.get(f"{field}_storage_key")
    url = payload.get(f"{field}_url")
    encoded = payload.get(f"{field}_base64")
    existing_path = payload.get(f"{field}_path")
    expected_size_bytes = payload.get(f"{field}_size_bytes")
    expected_sha256 = payload.get(f"{field}_sha256")

    if isinstance(spec, dict):
        storage_key = storage_key or spec.get("storage_key")
        url = url or spec.get("url")
        encoded = encoded or spec.get("base64")
        existing_path = existing_path or spec.get("path")
        expected_size_bytes = expected_size_bytes or spec.get("size_bytes")
        expected_sha256 = expected_sha256 or spec.get("sha256")
    elif isinstance(spec, str) and spec.startswith(("https://", "http://")):
        url = url or spec

    try:
        if storage_key:
            path = _download_storage_key(str(storage_key), destination)
        elif url:
            path = _download_url(
                str(url),
                destination,
                expected_size_bytes=(
                    int(expected_size_bytes) if expected_size_bytes is not None else None
                ),
                expected_sha256=(str(expected_sha256) if expected_sha256 else None),
            )
        elif encoded:
            try:
                raw = base64.b64decode(str(encoded), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"Invalid base64 supplied for {field}.") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            _validate_download(
                destination,
                content_type=None,
                expected_size_bytes=(
                    int(expected_size_bytes) if expected_size_bytes is not None else None
                ),
                expected_sha256=(str(expected_sha256) if expected_sha256 else None),
            )
            path = destination
        elif existing_path:
            source = Path(str(existing_path)).resolve()
            root = WORKSPACE.resolve()
            if root != source and root not in source.parents:
                raise ValueError(f"{field}_path must be inside {root}.")
            if not source.exists():
                raise FileNotFoundError(str(source))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            path = destination
        else:
            if required:
                raise ValueError(
                    f"{field}_url, {field}_storage_key, {field}_base64, or {field}_path is required."
                )
            return None
    except Exception as exc:
        if isinstance(exc, (InputTransferError, ValueError, FileNotFoundError)):
            raise type(exc)(f"Input '{field}' failed: {exc}") from exc
        raise InputTransferError(f"Input '{field}' failed: {exc}") from exc

    return path


def publish_file(
    path: Path,
    *,
    artist: str = "default",
    category: str = "outputs",
    ttl_seconds: int = 86400,
) -> dict:
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
            "sha256": _sha256(destination),
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
        "sha256": _sha256(path),
        "storage_key": key,
        "download_url": url,
        "expires_in_seconds": max(60, min(int(ttl_seconds), 604800)),
        "storage_mode": "s3_signed_urls",
    }


def publish_files(
    paths: Iterable[Path],
    *,
    artist: str = "default",
    category: str = "outputs",
    ttl_seconds: int = 86400,
) -> list[dict]:
    return [
        publish_file(
            Path(path),
            artist=artist,
            category=category,
            ttl_seconds=ttl_seconds,
        )
        for path in paths
    ]
