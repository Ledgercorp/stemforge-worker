"""Tests for storage readiness reporting.

`storage_info` used to derive everything from one environment variable, so it
reported `bucket_configured: true` and `signed_transfer_ready: true` while the
credentials were incomplete. Every upload then failed with
`PartialCredentialsError`, and the status action gave no hint why - it had
never tried to build a client.
"""

from __future__ import annotations

import json

import pytest

from app import storage


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "STEMFORGE_S3_BUCKET",
        "STEMFORGE_S3_ENDPOINT_URL",
        "STEMFORGE_S3_REGION",
        "STEMFORGE_S3_ACCESS_KEY_ID",
        "STEMFORGE_S3_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_unconfigured_reports_volume_mode():
    status = storage.storage_status()
    assert status["bucket_configured"] is False
    assert status["signed_transfer_ready"] is False
    assert status["mode"] == "runpod_volume"
    assert "client_error" not in status


def test_bucket_without_credentials_is_not_ready(monkeypatch):
    """The defect: a bucket name alone used to read as fully configured."""
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")

    # Named to match what botocore raises, without requiring it to be
    # installed: CI runs this suite without boto3.
    class PartialCredentialsError(Exception):
        pass

    def incomplete():
        raise PartialCredentialsError(
            "Partial credentials found in env, missing: secret_key"
        )

    monkeypatch.setattr(storage, "_s3_client", incomplete)
    status = storage.storage_status()

    # The bucket really is set, and saying otherwise would mislead too.
    assert status["bucket_configured"] is True
    # But nothing can be signed, so readiness must be false.
    assert status["signed_transfer_ready"] is False
    assert status["client_ready"] is False
    assert status["mode"] == "runpod_volume"
    assert "PartialCredentialsError" in status["client_error"]
    # The note names the variables an operator should check.
    assert "STEMFORGE_S3_ACCESS_KEY_ID" in status["note"]
    assert "STEMFORGE_S3_SECRET_ACCESS_KEY" in status["note"]


def test_working_client_reports_ready(monkeypatch):
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")
    monkeypatch.setattr(storage, "_s3_client", lambda: (object(), "stemforge"))

    status = storage.storage_status()
    assert status["signed_transfer_ready"] is True
    assert status["client_ready"] is True
    assert status["mode"] == "s3_signed_urls"
    assert "client_error" not in status
    assert status["note"] == "Private expiring transfer links are active."


def test_status_never_raises(monkeypatch):
    """A status probe must report a problem, not become one."""
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")

    def explode():
        raise RuntimeError("boto3 is not installed; S3 storage is unavailable.")

    monkeypatch.setattr(storage, "_s3_client", explode)
    status = storage.storage_status()
    assert status["signed_transfer_ready"] is False
    assert "boto3" in status["client_error"]


def test_configuration_report_names_missing_settings(monkeypatch):
    """Diagnosing a bad bucket must not mean guessing among five variables."""
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")
    monkeypatch.setenv("STEMFORGE_S3_REGION", "auto")
    monkeypatch.setattr(storage, "_s3_client", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    report = storage.storage_status()["configuration"]

    assert report["STEMFORGE_S3_BUCKET"]["set"] is True
    assert report["STEMFORGE_S3_BUCKET"]["empty"] is False
    # Absent credentials are visible as absent.
    assert report["STEMFORGE_S3_ACCESS_KEY_ID"]["set"] is False
    assert report["STEMFORGE_S3_SECRET_ACCESS_KEY"]["set"] is False


def test_configuration_report_never_discloses_a_credential(monkeypatch):
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")
    monkeypatch.setenv("STEMFORGE_S3_ACCESS_KEY_ID", "AKIAsecretvalue123")
    monkeypatch.setenv("STEMFORGE_S3_SECRET_ACCESS_KEY", "supersecretvalue456")
    monkeypatch.setattr(storage, "_s3_client", lambda: (object(), "stemforge"))

    status = storage.storage_status()
    serialized = json.dumps(status)

    assert "AKIAsecretvalue123" not in serialized
    assert "supersecretvalue456" not in serialized
    # But their shape is still reported, which is what finds a truncated paste.
    assert status["configuration"]["STEMFORGE_S3_ACCESS_KEY_ID"]["length"] == 18
    assert status["configuration"]["STEMFORGE_S3_SECRET_ACCESS_KEY"]["length"] == 19


def test_configuration_report_shows_non_secret_values(monkeypatch):
    """A wrong endpoint URL is only findable if you can see it."""
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")
    monkeypatch.setenv("STEMFORGE_S3_ENDPOINT_URL", "https://acct.r2.cloudflarestorage.com/stemforge")
    monkeypatch.setattr(storage, "_s3_client", lambda: (object(), "stemforge"))

    report = storage.storage_status()["configuration"]
    # The trailing bucket path is the classic mistake; it must be visible.
    assert report["STEMFORGE_S3_ENDPOINT_URL"]["value"].endswith("/stemforge")


def test_configuration_report_flags_stray_whitespace(monkeypatch):
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")
    monkeypatch.setenv("STEMFORGE_S3_ACCESS_KEY_ID", "abc123 ")
    monkeypatch.setattr(storage, "_s3_client", lambda: (object(), "stemforge"))

    report = storage.storage_status()["configuration"]
    assert report["STEMFORGE_S3_ACCESS_KEY_ID"]["has_surrounding_whitespace"] is True


def test_configuration_report_notices_aws_named_credentials(monkeypatch):
    """Pasting into AWS_* instead of STEMFORGE_S3_* is an easy mix-up."""
    monkeypatch.setenv("STEMFORGE_S3_BUCKET", "stemforge")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "abc123")
    monkeypatch.setattr(storage, "_s3_client", lambda: (object(), "stemforge"))

    report = storage.storage_status()["configuration"]
    assert report["AWS_ACCESS_KEY_ID"]["set"] is True
