"""Tests for the volume uploader.

These drive the real worker actions (`volume_upload_init`, `_chunk`,
`_finalize`) in-process by substituting the transport, so the chunking,
digest and verification protocol is exercised end to end without a network
call or an API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.api import handle_job
from tools import upload_audio


@pytest.fixture
def local_transport(monkeypatch):
    """Route the uploader's job submissions straight at the worker."""
    submitted: list[dict] = []

    def fake_post(endpoint: str, api_key: str, payload: dict) -> dict:
        submitted.append(payload)
        output = handle_job(payload)
        if output.get("status") == "rejected":
            raise upload_audio.UploadError(
                f"worker rejected the request: {output.get('reason')}"
            )
        return output

    monkeypatch.setattr(upload_audio, "_post", fake_post)
    return submitted


def _fixture(tmp_path: Path, name: str, size: int) -> Path:
    path = tmp_path / name
    # Deterministic, incompressible-ish bytes so chunk boundaries matter.
    path.write_bytes(bytes((i * 37 + 11) % 256 for i in range(size)))
    return path


def test_single_chunk_upload_round_trips(tmp_path, local_transport):
    source = _fixture(tmp_path, "clip.wav", 4096)
    result = upload_audio.upload(
        source, endpoint="test", api_key="test", chunk_bytes=1_000_000, quiet=True
    )

    assert result["status"] == "completed"
    assert result["sha256"] == upload_audio._digest(source)[0]
    assert result["size_bytes"] == 4096
    assert Path(result["path"]).read_bytes() == source.read_bytes()

    actions = [payload["action"] for payload in local_transport]
    assert actions == [
        "volume_upload_init",
        "volume_upload_chunk",
        "volume_upload_finalize",
    ]


def test_multi_chunk_upload_reassembles_in_order(tmp_path, local_transport):
    source = _fixture(tmp_path, "song.flac", 10_000)
    result = upload_audio.upload(
        source, endpoint="test", api_key="test", chunk_bytes=3_000, quiet=True
    )

    assert Path(result["path"]).read_bytes() == source.read_bytes()
    chunk_indexes = [
        payload["index"]
        for payload in local_transport
        if payload["action"] == "volume_upload_chunk"
    ]
    assert chunk_indexes == [0, 1, 2, 3]


def test_uploaded_path_is_inside_the_worker_workspace(tmp_path, local_transport):
    """The path must be usable as `audio_path`, which the worker sandboxes."""
    from app.storage import WORKSPACE

    source = _fixture(tmp_path, "clip.wav", 2048)
    result = upload_audio.upload(
        source, endpoint="test", api_key="test", chunk_bytes=1_000_000, quiet=True
    )
    resolved = Path(result["path"]).resolve()
    root = Path(WORKSPACE).resolve()
    assert root == resolved or root in resolved.parents


def test_upload_id_is_stable_for_identical_content(tmp_path):
    source = _fixture(tmp_path, "Take 1 (final).wav", 512)
    sha, _ = upload_audio._digest(source)
    first = upload_audio._upload_id(source, sha)
    second = upload_audio._upload_id(source, sha)
    assert first == second
    assert first.startswith("take-1--final-")
    assert all(c.isalnum() or c == "-" for c in first)


def test_digest_mismatch_is_reported(tmp_path, monkeypatch):
    """A finalize response that disagrees with the local digest must fail."""
    source = _fixture(tmp_path, "clip.wav", 1024)

    def lying_post(endpoint, api_key, payload):
        if payload["action"] == "volume_upload_finalize":
            return {"status": "completed", "path": "/tmp/x", "sha256": "0" * 64}
        return {"status": "completed"}

    monkeypatch.setattr(upload_audio, "_post", lying_post)
    with pytest.raises(upload_audio.UploadError, match="digest mismatch"):
        upload_audio.upload(
            source, endpoint="test", api_key="test", chunk_bytes=1_000_000, quiet=True
        )


def test_worker_rejects_a_corrupt_size_declaration(local_transport):
    """The worker's own verification must fail a wrong expected size."""
    output = handle_job(
        {
            "action": "volume_upload_init",
            "upload_id": "size-check-test",
            "filename": "clip.wav",
            "expected_size_bytes": 999,
            "expected_sha256": "",
            "expected_chunks": 1,
        }
    )
    assert output["status"] == "completed"
    handle_job(
        {
            "action": "volume_upload_chunk",
            "upload_id": "size-check-test",
            "index": 0,
            "data_base64": "AAAA",
        }
    )
    final = handle_job({"action": "volume_upload_finalize", "upload_id": "size-check-test"})
    assert final["status"] == "rejected"
    assert "Size verification failed" in final["reason"]


@pytest.mark.parametrize(
    "action,expected_field",
    [
        ("naturalize", "audio_path"),
        ("master_audio", "audio_path"),
        ("stem_remix", "master_path"),
        ("full_pass", "master_path"),
    ],
)
def test_emitted_job_targets_the_right_input_field(action, expected_field):
    import argparse

    args = argparse.Namespace(
        action=action, mode="quick", intensity=0.6, artist="sounddecay", song="test"
    )
    job = upload_audio.build_job("/runpod-volume/stemforge/transfers/x/song.wav", args)
    assert expected_field in job["input"]
    assert job["input"][expected_field].endswith("song.wav")
    assert job["policy"]["executionTimeout"] <= job["policy"]["ttl"]


def test_emitted_naturalize_job_passes_the_relay_validator(tmp_path):
    import argparse

    from tools import validate_jobs

    args = argparse.Namespace(
        action="naturalize", mode="quick", intensity=0.6, artist="sounddecay", song="test"
    )
    job = upload_audio.build_job("/runpod-volume/stemforge/transfers/x/song.wav", args)
    path = tmp_path / "naturalize-upload-20260805-1200.json"
    path.write_text(json.dumps(job), encoding="utf-8")

    issues = validate_jobs.validate_file(path)
    assert [i.render() for i in issues if i.level == validate_jobs.ERROR] == []


def test_cli_rejects_an_oversized_chunk_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test")
    source = _fixture(tmp_path, "clip.wav", 16)
    assert upload_audio.main([str(source), "--chunk-bytes", "8000000"]) == 2


def test_cli_requires_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    source = _fixture(tmp_path, "clip.wav", 16)
    assert upload_audio.main([str(source)]) == 2


def test_cli_reports_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test")
    assert upload_audio.main([str(tmp_path / "absent.wav")]) == 1
