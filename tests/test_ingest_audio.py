"""Tests for the storage ingest tool.

The transport is substituted so the whole flow - create_upload, the presigned
PUT, and the create_download readback - is exercised without network access,
an API key, or a configured bucket.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path

import pytest

from tools import ingest_audio


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes = b"", status: int = 200, headers: dict | None = None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def worker(monkeypatch):
    """Stand in for the worker's storage actions and the bucket itself."""
    state = {"objects": {}, "calls": [], "put_headers": None}

    def fake_run(endpoint, api_key, payload):
        state["calls"].append(payload["action"])
        if payload["action"] == "create_upload":
            key = f"incoming/{payload['artist']}/abc123/{payload['filename']}"
            return {
                "status": "completed",
                "storage_key": key,
                "upload_url": f"https://bucket.example/{key}?sig=x",
                "method": "PUT",
                "headers": {"Content-Type": payload["content_type"]},
            }
        if payload["action"] == "create_download":
            key = payload["storage_key"]
            if key not in state["objects"]:
                raise ingest_audio.IngestError("no such object")
            return {
                "status": "completed",
                "storage_key": key,
                "download_url": f"https://bucket.example/{key}?get=1",
            }
        raise AssertionError(f"unexpected action {payload['action']}")

    def fake_urlopen(request, timeout=None):
        url = request.full_url
        if request.get_method() == "PUT":
            key = url.split("https://bucket.example/")[1].split("?")[0]
            state["objects"][key] = request.data
            state["put_headers"] = dict(request.header_items())
            return _FakeResponse(status=200)
        if request.get_method() == "GET":
            # A presigned URL signs the method, so R2 answers 403 to anything
            # but the GET it was signed for. Model that: only a GET is served.
            key = url.split("https://bucket.example/")[1].split("?")[0]
            body = state["objects"][key]
            state["probe_headers"] = dict(request.header_items())
            return _FakeResponse(
                body[:1],
                status=206,
                headers={
                    "Content-Range": f"bytes 0-0/{len(body)}",
                    "Content-Length": "1",
                },
            )
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(ingest_audio, "_run", fake_run)
    monkeypatch.setattr(ingest_audio.urllib.request, "urlopen", fake_urlopen)
    return state


def test_local_file_round_trips(tmp_path, worker):
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF" + b"\x00" * 4096)

    record = ingest_audio.ingest(
        str(source), endpoint="test", api_key="test", quiet=True
    )

    assert record["storage_key"].endswith("master.wav")
    assert record["size_bytes"] == 4100
    assert worker["objects"][record["storage_key"]] == source.read_bytes()
    # create_upload, then the readback proof.
    assert worker["calls"] == ["create_upload", "create_download"]


def test_content_type_header_is_sent_with_the_put(tmp_path, worker):
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF")
    ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)
    headers = {k.lower(): v for k, v in (worker["put_headers"] or {}).items()}
    # A presigned PUT is signed over Content-Type; omitting it fails the signature.
    assert headers.get("Content-type") or headers.get("content-type")


def test_verification_can_be_skipped(tmp_path, worker):
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF")
    ingest_audio.ingest(
        str(source), endpoint="test", api_key="test", verify=False, quiet=True
    )
    assert worker["calls"] == ["create_upload"]


def test_size_mismatch_on_readback_is_reported(tmp_path, worker, monkeypatch):
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF" * 100)

    real_urlopen = ingest_audio.urllib.request.urlopen

    def truncated(request, timeout=None):
        if request.get_method() == "GET":
            return _FakeResponse(
                b"R", status=206, headers={"Content-Range": "bytes 0-0/1"}
            )
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(ingest_audio.urllib.request, "urlopen", truncated)
    with pytest.raises(ingest_audio.IngestError, match="expected"):
        ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)


def test_readback_probes_with_the_method_the_url_was_signed_for(tmp_path, worker):
    """The defect: a HEAD against a GET-presigned URL is a signature mismatch.

    R2 answered 403, which reads like a missing object and was not one - the
    bytes were already stored. The probe must use the signed method.
    """
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF" + b"\x00" * 4096)

    ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)

    headers = {k.lower(): v for k, v in worker["probe_headers"].items()}
    # One byte, not the whole 4 KB object, and certainly not a 200 MB master.
    assert headers["range"] == "bytes=0-0"


def test_readback_reports_a_genuinely_missing_object(tmp_path, worker, monkeypatch):
    """Fixing the false 403 must not stop a real 403 from being reported."""
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF")

    real_urlopen = ingest_audio.urllib.request.urlopen

    def forbidden(request, timeout=None):
        if request.get_method() == "GET":
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(ingest_audio.urllib.request, "urlopen", forbidden)
    with pytest.raises(ingest_audio.IngestError, match="not readable: HTTP 403"):
        ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)


def test_a_failed_verification_still_reports_the_key(tmp_path, worker, monkeypatch):
    """The bytes are already stored; losing the key orphans them for good.

    Nothing can list bucket objects through the worker API, so a key that is
    never reported cannot be rendered from, retried, or deleted. Two objects
    were stranded exactly this way before the readback was fixed.
    """
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF" * 100)

    real_urlopen = ingest_audio.urllib.request.urlopen

    def truncated(request, timeout=None):
        if request.get_method() == "GET":
            return _FakeResponse(
                b"R", status=206, headers={"Content-Range": "bytes 0-0/1"}
            )
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(ingest_audio.urllib.request, "urlopen", truncated)
    with pytest.raises(ingest_audio.IngestError, match=r"already stored at .*master\.wav"):
        ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)


DRIVE_URL = "https://drive.google.com/uc?export=download&id=FILEID"

# What Drive actually answers with for a file over roughly 100 MB.
CONFIRM_PAGE = b"""<!DOCTYPE html><html><body>
<p>Google Drive can't scan this file for viruses.</p>
<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
<input type="hidden" name="id" value="FILEID">
<input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t">
<input type="hidden" name="uuid" value="abc-123">
</form></body></html>"""


def _patch_opener(monkeypatch, handler):
    """Route the module's opener through `handler(url) -> _FakeResponse`."""
    class _Opener:
        def open(self, request, timeout=None):
            return handler(request.full_url)

    monkeypatch.setattr(
        ingest_audio.urllib.request, "build_opener", lambda *a, **k: _Opener()
    )


def test_html_quota_page_is_rejected_not_uploaded(worker, monkeypatch):
    """A Drive link over quota serves HTML; that must never reach the bucket."""
    _patch_opener(
        monkeypatch,
        lambda url: _FakeResponse(b"<!DOCTYPE html><html><body>Quota exceeded"),
    )
    with pytest.raises(ingest_audio.IngestError, match="over its download quota"):
        ingest_audio.ingest(
            DRIVE_URL, endpoint="test", api_key="test", name="master.wav", quiet=True,
        )
    assert worker["objects"] == {}


def test_a_large_file_confirmation_is_followed(worker, monkeypatch):
    """The defect: Sweet Sixteen's stems archive never downloaded.

    Drive will not serve a file over ~100 MB directly - it answers with a
    virus-scan interstitial carrying a confirmation form. The 49 MB stems
    downloaded fine and the archive of the same stems did not, which read as
    a broken link and was not one.
    """
    seen = []
    audio = b"RIFF" + b"\x00" * 4096

    def handler(url):
        seen.append(url)
        if url == DRIVE_URL:
            return _FakeResponse(CONFIRM_PAGE)
        return _FakeResponse(audio, headers={"Content-Disposition": 'attachment; filename="Stems.zip"'})

    _patch_opener(monkeypatch, handler)
    record = ingest_audio.ingest(
        DRIVE_URL, endpoint="test", api_key="test", quiet=True
    )

    assert record["size_bytes"] == len(audio)
    # The follow-up replays Google's own form, token and all.
    assert "drive.usercontent.google.com/download" in seen[1]
    assert "confirm=t" in seen[1] and "uuid=abc-123" in seen[1]
    # And the real filename is recovered rather than "download.bin".
    assert record["filename"] == "Stems.zip"


def test_confirmation_falls_back_to_the_file_id(worker, monkeypatch):
    """An interstitial without a parsable form must still be followed."""
    audio = b"RIFF" + b"\x00" * 64
    seen = []

    def handler(url):
        seen.append(url)
        if url == DRIVE_URL:
            return _FakeResponse(b"<!DOCTYPE html><html><body>needs confirming</body></html>")
        return _FakeResponse(audio)

    _patch_opener(monkeypatch, handler)
    ingest_audio.ingest(DRIVE_URL, endpoint="test", api_key="test",
                        name="stems.zip", quiet=True)

    assert "id=FILEID" in seen[1] and "confirm=t" in seen[1]


def test_an_explicit_name_still_wins_over_the_served_one(worker, monkeypatch):
    _patch_opener(
        monkeypatch,
        lambda url: _FakeResponse(
            b"RIFF", headers={"Content-Disposition": 'attachment; filename="wrong.wav"'}
        ),
    )
    record = ingest_audio.ingest(
        DRIVE_URL, endpoint="test", api_key="test", name="Master.wav", quiet=True
    )
    assert record["filename"] == "Master.wav"


def test_a_page_that_never_resolves_is_reported_not_stored(worker, monkeypatch):
    """Following the confirmation must not turn a failure into a stored page."""
    _patch_opener(
        monkeypatch,
        lambda url: _FakeResponse(b"<!DOCTYPE html><html><body>Sign in to continue"),
    )
    with pytest.raises(ingest_audio.IngestError, match="not shared"):
        ingest_audio.ingest(DRIVE_URL, endpoint="test", api_key="test",
                            name="x.wav", quiet=True)
    assert worker["objects"] == {}


def test_unconfigured_bucket_is_reported_actionably(tmp_path, monkeypatch):
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF")

    def unconfigured(endpoint, api_key, payload):
        # What the worker returns when STEMFORGE_S3_BUCKET is absent.
        raise ingest_audio.IngestError(
            "worker error: STEMFORGE_S3_BUCKET is not configured."
        )

    monkeypatch.setattr(ingest_audio, "_run", unconfigured)
    with pytest.raises(ingest_audio.IngestError, match="STEMFORGE_S3_BUCKET"):
        ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)


def test_missing_local_file_is_reported(worker):
    with pytest.raises(ingest_audio.IngestError, match="not found"):
        ingest_audio.ingest("/nope/missing.wav", endpoint="test", api_key="test", quiet=True)


def test_cli_requires_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    source = tmp_path / "a.wav"
    source.write_bytes(b"RIFF")
    assert ingest_audio.main([str(source)]) == 2


def test_cli_requires_a_source(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test")
    with pytest.raises(SystemExit):
        ingest_audio.main([])


def test_cli_prints_key_and_filename(tmp_path, worker, monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "test")
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF")

    assert ingest_audio.main([str(source), "--quiet"]) == 0
    out = capsys.readouterr().out.strip()
    key, filename = out.split("\t")
    assert filename == "master.wav"
    assert key.endswith("master.wav")


def test_ingested_key_is_usable_as_a_job_input(tmp_path, worker):
    """The whole point: the key must satisfy the relay contract."""
    import json

    from tools import validate_jobs

    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF")
    record = ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)

    job = {
        "input": {"action": "naturalize", "mode": "auto",
                  "audio_storage_key": record["storage_key"]},
        "policy": {"executionTimeout": 1_800_000, "ttl": 3_600_000},
    }
    path = tmp_path / "naturalize-ingested-20260805-1200.json"
    path.write_text(json.dumps(job), encoding="utf-8")

    errors = [i for i in validate_jobs.validate_file(path) if i.level == validate_jobs.ERROR]
    assert errors == []


def _zip_bytes(names: dict[str, bytes]) -> bytes:
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in names.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_expand_stores_one_object_per_stem(tmp_path, worker):
    """A single stems archive becomes one storage key per stem."""
    source = tmp_path / "stems.zip"
    source.write_bytes(
        _zip_bytes({
            "Lead Vocals.wav": b"RIFF" + b"\x01" * 64,
            "Drums.wav": b"RIFF" + b"\x02" * 64,
            "Bass.wav": b"RIFF" + b"\x03" * 64,
            "notes.txt": b"ignore me",
        })
    )

    records = ingest_audio.ingest_many(
        str(source), endpoint="test", api_key="test", expand=True, quiet=True
    )

    assert len(records) == 3
    assert {r["filename"] for r in records} == {"Lead Vocals.wav", "Drums.wav", "Bass.wav"}
    # Each stem is a distinct object, not the archive.
    assert len({r["storage_key"] for r in records}) == 3


def test_expand_leaves_a_non_archive_alone(tmp_path, worker):
    source = tmp_path / "master.wav"
    source.write_bytes(b"RIFF" + b"\x00" * 32)
    records = ingest_audio.ingest_many(
        str(source), endpoint="test", api_key="test", expand=True, quiet=True
    )
    assert len(records) == 1
    assert records[0]["filename"] == "master.wav"


def test_without_expand_an_archive_is_stored_whole(tmp_path, worker):
    """MIDI archives must stay zipped: the worker wants midi_zip intact."""
    source = tmp_path / "midi.zip"
    source.write_bytes(_zip_bytes({"Drums.mid": b"MThd", "Bass.mid": b"MThd"}))
    records = ingest_audio.ingest_many(
        str(source), endpoint="test", api_key="test", expand=False, quiet=True
    )
    assert len(records) == 1
    assert records[0]["filename"] == "midi.zip"


def test_expand_rejects_a_truncated_archive(tmp_path, worker):
    source = tmp_path / "stems.zip"
    raw = _zip_bytes({"A.wav": b"RIFF" * 40, "B.wav": b"RIFF" * 40, "C.wav": b"RIFF" * 40})
    source.write_bytes(raw[150:])
    with pytest.raises(ingest_audio.IngestError, match="corrupt"):
        ingest_audio.ingest_many(
            str(source), endpoint="test", api_key="test", expand=True, quiet=True
        )
    assert worker["objects"] == {}
