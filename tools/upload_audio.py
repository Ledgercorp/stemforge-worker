#!/usr/bin/env python3
"""Upload a local audio file straight to the StemForge RunPod volume.

Renders need their input reachable by the worker. Without S3 configured, the
options are a public URL, inline base64, or the volume. URLs expire and
inline base64 bloats git history, so for real music files the volume is the
right home: uploaded once, addressed by a stable path, reusable across every
render of that song.

This drives the worker's own `volume_upload_init` / `volume_upload_chunk` /
`volume_upload_finalize` actions over the public job API, so it needs nothing
but an endpoint id and an API key with run scope — the same key the relay
already uses.

    export RUNPOD_API_KEY=...
    python tools/upload_audio.py mysong.wav
    python tools/upload_audio.py mysong.wav --emit-job jobs/mysong-20260805-1200.json \
        --action naturalize --mode quick --intensity 0.6

The finalized path is printed on the last line and is what goes into a job as
`audio_path`, `master_path` or a stem's `path`.

Uploads are verified end to end: size and SHA-256 are computed locally, sent
with the init call, and re-checked by the worker on finalize. A mismatch
deletes the assembled file rather than leaving a corrupt input in place.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "dyup1dztjr4u15"
API_ROOT = "https://api.runpod.ai/v2"

# The worker rejects chunks over 7,000,000 raw bytes. Stay clear of the limit:
# base64 inflates by 4/3 and the request carries JSON overhead on top.
CHUNK_BYTES = 5_000_000

POLL_SECONDS = 3
POLL_ATTEMPTS = 200
TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".ogg", ".opus"}


class UploadError(RuntimeError):
    """Raised when the endpoint refuses or fails an upload step."""


def _post(endpoint: str, api_key: str, payload: dict) -> dict:
    """Submit one job and poll it to a terminal state."""
    request = urllib.request.Request(
        f"{API_ROOT}/{endpoint}/run",
        data=json.dumps({"input": payload, "policy": {"ttl": 3_600_000}}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            submitted = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise UploadError(f"submit failed: HTTP {exc.code}") from exc

    job_id = submitted.get("id")
    if not job_id:
        raise UploadError(f"submit returned no job id: {submitted}")

    status_request = urllib.request.Request(
        f"{API_ROOT}/{endpoint}/status/{job_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    for _ in range(POLL_ATTEMPTS):
        time.sleep(POLL_SECONDS)
        try:
            with urllib.request.urlopen(status_request, timeout=60) as response:
                status = json.loads(response.read())
        except urllib.error.HTTPError:
            continue  # transient; keep polling
        if status.get("status") in TERMINAL:
            break
    else:
        raise UploadError(f"job {job_id} never reached a terminal state")

    if status.get("status") != "COMPLETED":
        raise UploadError(f"job {job_id} finished {status.get('status')}: {status}")

    output = status.get("output") or {}
    if output.get("status") == "rejected":
        raise UploadError(f"worker rejected the request: {output.get('reason')}")
    return output


def _digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
            size += len(block)
    return sha.hexdigest(), size


def _upload_id(path: Path, sha256: str) -> str:
    stem = "".join(c if c.isalnum() else "-" for c in path.stem).strip("-").lower()
    return f"{stem[:48] or 'upload'}-{sha256[:12]}"


def upload(
    path: Path,
    *,
    endpoint: str,
    api_key: str,
    chunk_bytes: int = CHUNK_BYTES,
    quiet: bool = False,
) -> dict:
    """Upload one file and return the worker's finalize response."""
    sha256, size = _digest(path)
    chunks = max(1, -(-size // chunk_bytes))
    upload_id = _upload_id(path, sha256)

    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    log(f"{path.name}: {size:,} bytes, sha256 {sha256[:16]}…, {chunks} chunk(s)")

    _post(
        endpoint,
        api_key,
        {
            "action": "volume_upload_init",
            "upload_id": upload_id,
            "filename": path.name,
            "expected_size_bytes": size,
            "expected_sha256": sha256,
            "expected_chunks": chunks,
        },
    )

    with path.open("rb") as handle:
        for index in range(chunks):
            block = handle.read(chunk_bytes)
            _post(
                endpoint,
                api_key,
                {
                    "action": "volume_upload_chunk",
                    "upload_id": upload_id,
                    "index": index,
                    "data_base64": base64.b64encode(block).decode(),
                },
            )
            log(f"  chunk {index + 1}/{chunks} ({len(block):,} bytes)")

    result = _post(
        endpoint,
        api_key,
        {"action": "volume_upload_finalize", "upload_id": upload_id},
    )
    if result.get("sha256") != sha256:
        raise UploadError(
            f"digest mismatch after finalize: local {sha256}, remote {result.get('sha256')}"
        )
    log(f"verified: {result.get('path')}")
    return result


def build_job(volume_path: str, args: argparse.Namespace) -> dict:
    """Assemble a ready-to-commit job file around an uploaded input."""
    payload: dict = {"action": args.action}
    field = "master" if args.action in ("stem_remix", "full_pass") else "audio"
    payload[f"{field}_path"] = volume_path
    if args.artist:
        payload["artist"] = args.artist
    if args.song:
        payload["song"] = args.song
    if args.action == "naturalize":
        payload["mode"] = args.mode
        payload["intensity"] = args.intensity
        payload["passes"] = 1
        payload["retain_original"] = True
        payload["retain_pre_denoise"] = True
    return {
        "input": payload,
        "policy": {"executionTimeout": 1_800_000, "ttl": 3_600_000},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload audio to the StemForge RunPod volume."
    )
    parser.add_argument("files", nargs="+", type=Path, help="Local audio files.")
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT))
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=CHUNK_BYTES,
        help=f"Bytes per chunk before base64 (default {CHUNK_BYTES:,}; worker limit 7,000,000).",
    )
    parser.add_argument("--emit-job", type=Path, help="Write a job file for the first upload.")
    parser.add_argument("--action", default="naturalize", help="Action for --emit-job.")
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--artist", default="sounddecay")
    parser.add_argument("--song")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print(
            "RUNPOD_API_KEY is not set. Export a RunPod key with run scope; the\n"
            "same key the relay uses is sufficient.",
            file=sys.stderr,
        )
        return 2

    missing = [str(path) for path in args.files if not path.is_file()]
    if missing:
        for path in missing:
            print(f"not found: {path}", file=sys.stderr)
        return 1

    if args.chunk_bytes > 7_000_000:
        print("--chunk-bytes exceeds the worker's 7,000,000 byte limit.", file=sys.stderr)
        return 2

    results = []
    for path in args.files:
        if path.suffix.lower() not in AUDIO_SUFFIXES and not args.quiet:
            print(f"note: {path.name} has an unusual audio suffix", file=sys.stderr)
        try:
            results.append(
                upload(
                    path,
                    endpoint=args.endpoint,
                    api_key=api_key,
                    chunk_bytes=args.chunk_bytes,
                    quiet=args.quiet,
                )
            )
        except UploadError as exc:
            print(f"{path.name}: {exc}", file=sys.stderr)
            return 1

    if args.emit_job:
        job = build_job(str(results[0]["path"]), args)
        args.emit_job.parent.mkdir(parents=True, exist_ok=True)
        args.emit_job.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.emit_job}", file=sys.stderr)

    for result in results:
        print(result["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
