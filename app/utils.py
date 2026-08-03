from __future__ import annotations

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


def download_file(url: str, destination: Path, timeout: int = 900) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

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
