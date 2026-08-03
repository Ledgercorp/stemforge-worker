from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests


WORKSPACE = Path(os.environ.get("STEMFORGE_WORKSPACE", "/tmp/stemforge"))
WORKSPACE.mkdir(parents=True, exist_ok=True)


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower()).strip("'")


def safe_filename_from_url(url: str, fallback: str = "audio.wav") -> str:
    name = Path(urlparse(url).path).name
    if not name:
        name = fallback
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name or fallback


def _decode_data_url(url: str, destination: Path) -> Path:
    if "," not in url:
        raise ValueError("Malformed data URL.")

    header, encoded = url.split(",", 1)
    if ";base64" not in header.lower():
        raise ValueError("Only base64 data URLs are supported.")

    encoded = "".join(encoded.split())
    if len(encoded) > 9_000_000:
        raise ValueError("Embedded audio exceeds the safe request-size limit.")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Embedded audio contains invalid base64.") from exc

    if not payload:
        raise ValueError("Embedded audio decoded to an empty file.")

    destination.write_bytes(payload)
    return destination


def download_file(url: str, destination: Path, timeout: int = 900) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if url.startswith("data:"):
        return _decode_data_url(url, destination)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    if destination.stat().st_size == 0:
        raise ValueError("Downloaded file is empty.")

    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporary_job_dir(prefix: str = "job"):
    return tempfile.TemporaryDirectory(prefix=f"stemforge-{prefix}-")
