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
        if request.get_method() == "HEAD":
            key = url.split("https://bucket.example/")[1].split("?")[0]
            body = state["objects"][key]
            return _FakeResponse(headers={"Content-Length": str(len(body))})
        raise AssertionError(f"unexpected request {request.get_method()} {url}")

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
        if request.get_method() == "HEAD":
            return _FakeResponse(headers={"Content-Length": "1"})
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(ingest_audio.urllib.request, "urlopen", truncated)
    with pytest.raises(ingest_audio.IngestError, match="expected"):
        ingest_audio.ingest(str(source), endpoint="test", api_key="test", quiet=True)


def test_html_quota_page_is_rejected_not_uploaded(worker, monkeypatch):
    """A Drive link over quota serves HTML; that must never reach the bucket."""
    def html(request, timeout=None):
        return _FakeResponse(b"<!DOCTYPE html><html><body>Quota exceeded")

    monkeypatch.setattr(ingest_audio.urllib.request, "urlopen", html)
    with pytest.raises(ingest_audio.IngestError, match="HTML page"):
        ingest_audio.ingest(
            "https://drive.google.com/uc?export=download&id=x",
            endpoint="test", api_key="test", name="master.wav", quiet=True,
        )
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
