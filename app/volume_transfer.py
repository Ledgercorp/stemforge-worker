from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import shutil
from pathlib import Path

from app.storage import WORKSPACE

TRANSFER_ROOT = WORKSPACE / "transfers"
TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)
MAX_CHUNK_BYTES = 7_000_000
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


def _safe_id(value: object, field: str = "upload_id") -> str:
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise ValueError(f"{field} must use 1-80 letters, numbers, dot, underscore, or hyphen.")
    return text


def _upload_dir(upload_id: str) -> Path:
    return TRANSFER_ROOT / _safe_id(upload_id)


def volume_upload_init_job(payload: dict) -> dict:
    upload_id = _safe_id(payload.get("upload_id"))
    filename = str(payload.get("filename") or "input.bin").strip()
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._-") or "input.bin"
    expected_size = int(payload.get("expected_size_bytes") or 0)
    expected_sha256 = str(payload.get("expected_sha256") or "").lower().strip()
    expected_chunks = int(payload.get("expected_chunks") or 0)
    if expected_size <= 0 or expected_chunks <= 0:
        return {"status": "rejected", "reason": "expected_size_bytes and expected_chunks must be positive."}
    if expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        return {"status": "rejected", "reason": "expected_sha256 must be a 64-character lowercase hex digest."}
    root = _upload_dir(upload_id)
    if root.exists():
        shutil.rmtree(root)
    (root / "chunks").mkdir(parents=True, exist_ok=True)
    manifest = {
        "upload_id": upload_id,
        "filename": filename,
        "expected_size_bytes": expected_size,
        "expected_sha256": expected_sha256,
        "expected_chunks": expected_chunks,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "completed", **manifest, "transfer_root": str(root)}


def volume_upload_chunk_job(payload: dict) -> dict:
    upload_id = _safe_id(payload.get("upload_id"))
    index = int(payload.get("index"))
    encoded = str(payload.get("data_base64") or "")
    if index < 0:
        return {"status": "rejected", "reason": "index must be non-negative."}
    root = _upload_dir(upload_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"status": "rejected", "reason": "Upload session has not been initialized."}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if index >= int(manifest["expected_chunks"]):
        return {"status": "rejected", "reason": "Chunk index exceeds expected_chunks."}
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("data_base64 is invalid.") from exc
    if not raw:
        return {"status": "rejected", "reason": "Chunk is empty."}
    if len(raw) > MAX_CHUNK_BYTES:
        return {"status": "rejected", "reason": f"Chunk exceeds {MAX_CHUNK_BYTES} bytes."}
    chunk_path = root / "chunks" / f"{index:06d}.part"
    chunk_path.write_bytes(raw)
    return {
        "status": "completed",
        "upload_id": upload_id,
        "index": index,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def volume_upload_finalize_job(payload: dict) -> dict:
    upload_id = _safe_id(payload.get("upload_id"))
    root = _upload_dir(upload_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"status": "rejected", "reason": "Upload session has not been initialized."}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_chunks = int(manifest["expected_chunks"])
    parts = [root / "chunks" / f"{index:06d}.part" for index in range(expected_chunks)]
    missing = [path.name for path in parts if not path.exists()]
    if missing:
        return {"status": "rejected", "reason": "Missing chunks.", "missing": missing[:20], "missing_count": len(missing)}
    final_path = root / manifest["filename"]
    digest = hashlib.sha256()
    total = 0
    with final_path.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    total += len(block)
    actual_sha = digest.hexdigest()
    if total != int(manifest["expected_size_bytes"]):
        final_path.unlink(missing_ok=True)
        return {"status": "rejected", "reason": "Size verification failed.", "expected": manifest["expected_size_bytes"], "actual": total}
    expected_sha = str(manifest.get("expected_sha256") or "")
    if expected_sha and actual_sha != expected_sha:
        final_path.unlink(missing_ok=True)
        return {"status": "rejected", "reason": "SHA-256 verification failed.", "expected": expected_sha, "actual": actual_sha}
    shutil.rmtree(root / "chunks", ignore_errors=True)
    return {
        "status": "completed",
        "upload_id": upload_id,
        "filename": manifest["filename"],
        "path": str(final_path),
        "size_bytes": total,
        "sha256": actual_sha,
    }


def _safe_workspace_path(value: object) -> Path:
    path = Path(str(value or "")).resolve()
    workspace = WORKSPACE.resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError(f"path must be inside {workspace}.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def volume_file_info_job(payload: dict) -> dict:
    path = _safe_workspace_path(payload.get("path"))
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"status": "completed", "path": str(path), "filename": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def volume_file_chunk_job(payload: dict) -> dict:
    path = _safe_workspace_path(payload.get("path"))
    offset = max(0, int(payload.get("offset") or 0))
    length = max(1, min(int(payload.get("length") or 4_000_000), 5_500_000))
    size = path.stat().st_size
    if offset >= size:
        return {"status": "completed", "path": str(path), "offset": offset, "data_base64": "", "bytes_read": 0, "eof": True}
    with path.open("rb") as source:
        source.seek(offset)
        raw = source.read(length)
    return {
        "status": "completed",
        "path": str(path),
        "offset": offset,
        "bytes_read": len(raw),
        "eof": offset + len(raw) >= size,
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def volume_delete_job(payload: dict) -> dict:
    paths = payload.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    deleted = []
    workspace = WORKSPACE.resolve()
    for value in paths:
        path = Path(str(value)).resolve()
        if path == workspace or workspace not in path.parents:
            raise ValueError(f"Refusing to delete outside {workspace}.")
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            deleted.append(str(path))
        elif path.exists():
            path.unlink()
            deleted.append(str(path))
    return {"status": "completed", "deleted": deleted}
