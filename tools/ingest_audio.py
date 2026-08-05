#!/usr/bin/env python3
"""Ingest audio into StemForge storage once, then reference it forever.

Renders currently take their input from Google Drive links, which expire and
can serve an HTML quota page instead of audio - a failure that surfaces much
later as a confusing decode error. This ingests a file once into the
S3-compatible bucket and returns a stable `storage_key` that every later
render can use.

The worker does the signing: `create_upload` returns a presigned PUT URL, the
bytes go straight to the bucket, and `create_download` proves the object is
readable afterwards. Nothing but the RunPod job API is needed, so the same
run-scoped key the relay already uses is sufficient.

    export RUNPOD_API_KEY=...
    python tools/ingest_audio.py ~/Music/master.wav
    python tools/ingest_audio.py --from-url "https://drive.google.com/uc?..." --name master.wav

Each ingested file prints one line: `<storage_key>\t<filename>`. Use the key
as `audio_storage_key`, `master_storage_key`, or a stem's `storage_key`.

Requires S3 storage to be configured on the endpoint. Without it the worker
raises "STEMFORGE_S3_BUCKET is not configured" and this reports that plainly
rather than failing obscurely.
"""

from __future__ import annotations

import argparse
import html
import http.cookiejar
import io
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Sequence

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".ogg", ".opus"}

DEFAULT_ENDPOINT = "dyup1dztjr4u15"
API_ROOT = "https://api.runpod.ai/v2"
UPLOAD_TTL_SECONDS = 3600
POLL_SECONDS = 3
POLL_ATTEMPTS = 100
TERMINAL = {"COMPLETED", "FAILED", "TIMED_OUT", "CANCELLED"}


class IngestError(RuntimeError):
    """Raised when the endpoint or the bucket refuses a step."""


def _run(endpoint: str, api_key: str, payload: dict) -> dict:
    """Submit one job to the worker and poll it to a terminal state."""
    request = urllib.request.Request(
        f"{API_ROOT}/{endpoint}/run",
        data=json.dumps({"input": payload, "policy": {"ttl": 600_000}}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            submitted = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise IngestError(f"submit failed: HTTP {exc.code}") from exc

    job_id = submitted.get("id")
    if not job_id:
        raise IngestError(f"submit returned no job id: {submitted}")

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
            continue
        if status.get("status") in TERMINAL:
            break
    else:
        raise IngestError(f"job {job_id} never reached a terminal state")

    if status.get("status") != "COMPLETED":
        raise IngestError(f"job {job_id} finished {status.get('status')}")

    output = status.get("output") or {}
    if output.get("status") == "rejected":
        raise IngestError(f"worker rejected the request: {output.get('reason')}")
    if output.get("status") == "error":
        detail = output.get("error") or output.get("error_type") or "no detail"
        if "STEMFORGE_S3_BUCKET" in str(detail):
            raise IngestError(
                "S3 storage is not configured on the endpoint. Set "
                "STEMFORGE_S3_BUCKET and the S3 credentials in the RunPod "
                "template; see docs/DEPLOYMENT.md."
            )
        raise IngestError(f"worker error: {detail}")
    return output


def _looks_like_html(data: bytes) -> bool:
    return data[:512].lstrip().lower().startswith((b"<!doctype html", b"<html"))


def _filename_from_headers(headers) -> str:
    """Pull the real filename out of Content-Disposition, if it carries one."""
    disposition = headers.get("Content-Disposition") or ""
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition)
    if not match:
        return ""
    return Path(urllib.parse.unquote(match.group(1).strip())).name


def _drive_confirm_url(page: bytes, source: str) -> str:
    """Build the follow-up URL for Drive's large-file confirmation page.

    Drive does not serve a file over roughly 100 MB directly: it answers with
    an interstitial saying it cannot scan the file for viruses, carrying a
    form that repeats the request with a confirmation token. Small files skip
    this entirely, which is why a set of 49 MB stems downloads cleanly and a
    single archive of the same stems does not.
    """
    text = page.decode("utf-8", errors="replace")

    # Preferred: replay the form the page actually contains, so the token and
    # any per-request uuid come from Google rather than from a guess.
    form = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
    if form:
        action = html.unescape(form.group(1))
        fields = dict(
            (html.unescape(n), html.unescape(v))
            for n, v in re.findall(
                r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text
            )
        )
        if fields:
            return f"{action}?{urllib.parse.urlencode(fields)}"

    # Fallback: the documented direct form, which needs only the file id.
    file_id = ""
    query = urllib.parse.parse_qs(urllib.parse.urlparse(source).query)
    if query.get("id"):
        file_id = query["id"][0]
    else:
        path_match = re.search(r"/file/d/([^/]+)", source)
        if path_match:
            file_id = path_match.group(1)
    if not file_id:
        return ""
    return (
        "https://drive.usercontent.google.com/download?"
        + urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    )


def _fetch(url: str, opener) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "stemforge-ingest"})
    try:
        with opener.open(request, timeout=900) as response:
            return response.read(), _filename_from_headers(response.headers)
    except urllib.error.HTTPError as exc:
        raise IngestError(f"could not fetch {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise IngestError(f"could not fetch {url}: {exc.reason}") from exc


def _read_source(source: str, name: str | None) -> tuple[bytes, str]:
    """Return the bytes to ingest and the filename to record."""
    if source.startswith(("http://", "https://")):
        # Drive sets a cookie on the interstitial and expects it back on the
        # confirmed request, so both go through one opener.
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        data, served_name = _fetch(source, opener)

        if _looks_like_html(data):
            confirm_url = _drive_confirm_url(data, source)
            if confirm_url:
                data, confirmed_name = _fetch(confirm_url, opener)
                served_name = confirmed_name or served_name

        if _looks_like_html(data):
            lowered = data[:4096].lower()
            if b"quota" in lowered:
                reason = "the file is over its download quota"
            elif b"sign in" in lowered or b"signin" in lowered:
                reason = "the file is not shared with anyone holding the link"
            else:
                reason = (
                    "the confirmation step did not yield the file; check the "
                    "link is shared with anyone holding it"
                )
            raise IngestError(
                f"{source} returned an HTML page rather than audio: {reason}."
            )
        return data, name or served_name or "download.bin"

    path = Path(source)
    if not path.is_file():
        raise IngestError(f"not found: {source}")
    return path.read_bytes(), name or path.name


def ingest(
    source: str,
    *,
    endpoint: str,
    api_key: str,
    name: str | None = None,
    artist: str = "sounddecay",
    verify: bool = True,
    quiet: bool = False,
) -> dict:
    """Ingest one file and return the worker's upload record."""

    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    data, filename = _read_source(source, name)
    return _ingest_bytes(
        data, filename, endpoint=endpoint, api_key=api_key,
        artist=artist, verify=verify, quiet=quiet,
    )


def _ingest_bytes(
    data: bytes,
    filename: str,
    *,
    endpoint: str,
    api_key: str,
    artist: str = "sounddecay",
    verify: bool = True,
    quiet: bool = False,
) -> dict:
    """Store bytes already in hand and return the upload record."""

    def log(message: str) -> None:
        if not quiet:
            print(message, file=sys.stderr, flush=True)

    log(f"{filename}: {len(data):,} bytes")

    created = _run(
        endpoint,
        api_key,
        {
            "action": "create_upload",
            "filename": filename,
            "artist": artist,
            "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
            "ttl_seconds": UPLOAD_TTL_SECONDS,
        },
    )
    upload_url = created.get("upload_url")
    storage_key = created.get("storage_key")
    if not upload_url or not storage_key:
        raise IngestError(f"create_upload returned no URL or key: {created}")

    headers = dict(created.get("headers") or {})
    request = urllib.request.Request(upload_url, data=data, method="PUT", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            if response.status not in (200, 201, 204):
                raise IngestError(f"upload returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise IngestError(f"upload failed: HTTP {exc.code} {exc.reason}") from exc
    log(f"  uploaded to {storage_key}")

    if verify:
        # Prove the object is readable before anyone builds a render on it.
        # Any failure past this point must still name the key: the bytes are
        # already in the bucket, and a key nobody recorded cannot be rendered
        # from, retried, or deleted - the API has no way to list objects.
        try:
            readback = _run(
                endpoint,
                api_key,
                {"action": "create_download", "storage_key": storage_key, "ttl_seconds": 600},
            )
            download_url = readback.get("download_url")
            if not download_url:
                raise IngestError(f"create_download returned no URL: {readback}")
            length = _stored_size(download_url)
            if length and length != len(data):
                raise IngestError(
                    f"stored object is {length} bytes, expected {len(data)}"
                )
        except IngestError as exc:
            raise IngestError(f"{exc} (already stored at {storage_key})") from exc
        log("  verified readable")

    return {"storage_key": storage_key, "filename": filename, "size_bytes": len(data)}


def _stored_size(download_url: str) -> int:
    """Return the size of a stored object, reading one byte to find out.

    The probe has to be a GET. A presigned URL signs the HTTP method into the
    signature, so sending HEAD to a URL signed for `get_object` is a signature
    mismatch - Cloudflare R2 answers 403, which reads exactly like a missing
    object and is not one. A `Range: bytes=0-0` request uses the signed method
    and comes back as a 206 whose Content-Range carries the full length, so
    the object is proven readable without downloading it.
    """
    probe = urllib.request.Request(download_url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(probe, timeout=120) as response:
            content_range = response.headers.get("Content-Range") or ""
            length_header = response.headers.get("Content-Length") or "0"
    except urllib.error.HTTPError as exc:
        raise IngestError(f"stored object is not readable: HTTP {exc.code}") from exc

    if content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        # A server may answer `bytes 0-0/*` when it will not state the total.
        return int(total) if total.isdigit() else 0
    # No Content-Range means the range was ignored and this is the whole body.
    # The object is readable either way, which is what the probe is for.
    return int(length_header) if length_header.isdigit() else 0


def _archive_members(data: bytes, filename: str = "") -> list[tuple[str, bytes]]:
    """Return the audio members of a zip, or [] when this is not one.

    Detection uses the magic bytes *and* the filename: an archive truncated
    from the front no longer starts with PK, and gating on magic alone would
    quietly store the damaged bytes as a single object instead of reporting
    them.
    """
    if data[:2] != b"PK" and not filename.lower().endswith(".zip"):
        return []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        label = filename or "input"
        raise IngestError(
            f"'{label}' is a corrupt or truncated zip archive ({exc}). "
            "Re-upload it."
        )
    # A front-truncated archive lists members but cannot read them; catch that
    # here rather than storing a set of empty objects.
    for item in archive.infolist():
        if item.header_offset < 0 or item.header_offset >= len(data):
            raise IngestError(
                "archive is truncated or corrupt: "
                f"'{item.filename}' is declared at offset {item.header_offset} "
                f"in a {len(data)} byte file. Re-upload it."
            )
    members = [
        item
        for item in archive.infolist()
        if not item.is_dir()
        and Path(item.filename).suffix.lower() in AUDIO_SUFFIXES
        and not Path(item.filename).name.startswith("._")
    ]
    return [(Path(item.filename).name, archive.read(item)) for item in members]


def ingest_many(
    source: str,
    *,
    endpoint: str,
    api_key: str,
    name: str | None = None,
    artist: str = "sounddecay",
    verify: bool = True,
    expand: bool = False,
    quiet: bool = False,
) -> list[dict]:
    """Ingest one source, expanding an archive of audio when asked."""
    if not expand:
        return [
            ingest(
                source, endpoint=endpoint, api_key=api_key, name=name,
                artist=artist, verify=verify, quiet=quiet,
            )
        ]

    data, filename = _read_source(source, name)
    members = _archive_members(data, filename)
    if not members:
        # Not an archive, or an archive with no audio: treat it as one file.
        return [
            _ingest_bytes(
                data, filename, endpoint=endpoint, api_key=api_key,
                artist=artist, verify=verify, quiet=quiet,
            )
        ]

    if not quiet:
        print(f"{filename}: expanding {len(members)} audio member(s)", file=sys.stderr)
    return [
        _ingest_bytes(
            member_bytes, member_name, endpoint=endpoint, api_key=api_key,
            artist=artist, verify=verify, quiet=quiet,
        )
        for member_name, member_bytes in members
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest audio into StemForge storage and print stable keys."
    )
    parser.add_argument("sources", nargs="*", help="Local paths, or URLs with --from-url.")
    parser.add_argument("--from-url", action="append", default=[], help="A URL to ingest.")
    parser.add_argument("--name", help="Filename to record (single source only).")
    parser.add_argument("--artist", default="sounddecay")
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT))
    parser.add_argument("--no-verify", action="store_true", help="Skip the readback check.")
    parser.add_argument(
        "--expand",
        action="store_true",
        help="Expand a zip of audio into one stored object per member.",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="Emit a JSON record.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    sources = list(args.sources) + list(args.from_url)
    if not sources:
        parser.error("supply at least one local path or --from-url")
    if args.name and len(sources) != 1:
        parser.error("--name applies to a single source")

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("RUNPOD_API_KEY is not set.", file=sys.stderr)
        return 2

    records = []
    for source in sources:
        try:
            records.extend(
                ingest_many(
                    source,
                    endpoint=args.endpoint,
                    api_key=api_key,
                    name=args.name,
                    artist=args.artist,
                    verify=not args.no_verify,
                    expand=args.expand,
                    quiet=args.quiet,
                )
            )
        except IngestError as exc:
            print(f"{source}: {exc}", file=sys.stderr)
            return 1

    if args.as_json:
        print(json.dumps({"ingested": records}, indent=2))
    else:
        for record in records:
            print(f"{record['storage_key']}\t{record['filename']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
