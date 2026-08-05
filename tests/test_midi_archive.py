"""Regression tests for MIDI archive handling in the full pass.

A zip's central directory lives at the end of the file, so an archive
truncated from the front still lists its members correctly and looks intact.
Reading one then seeks to an offset that does not exist and raises
`OSError(Errno 22)` with no message, filename or cause. In production that
surfaced as an expensive full pass dying thirty seconds in, twice, with
`{"status": "error", "error_type": "OSError", "error": null}` and nothing
pointing at the MIDI input.

The archive committed to this repository is exactly that shape: a directory
declaring seven members totalling 18,700 compressed bytes inside a 4,970 byte
file, leaving five members at negative offsets.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.full_pass_v2 import _safe_extract_midi


def _midi_bytes(note: int = 60) -> bytes:
    """A minimal but valid standard MIDI file."""
    track = bytes(
        [
            0x00, 0x90, note, 0x40,  # note on
            0x60, 0x80, note, 0x40,  # note off
            0x00, 0xFF, 0x2F, 0x00,  # end of track
        ]
    )
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + (96).to_bytes(2, "big")
    return header + b"MTrk" + len(track).to_bytes(4, "big") + track


def _archive(tmp_path: Path, names: list[str]) -> Path:
    path = tmp_path / "midi.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(names):
            archive.writestr(name, _midi_bytes(60 + index))
    return path


def test_valid_archive_extracts(tmp_path):
    archive = _archive(tmp_path, ["Song (Drums).mid", "Song (Bass).mid"])
    extracted = _safe_extract_midi(archive, tmp_path / "out")
    assert len(extracted) == 2
    assert all(path.exists() and path.stat().st_size > 0 for path in extracted)
    # Names with spaces and parentheses survive, as real exports use them.
    assert {path.name for path in extracted} == {"Song (Drums).mid", "Song (Bass).mid"}


def test_front_truncated_archive_is_reported_clearly(tmp_path):
    """The production defect: listable, unreadable, and previously a bare OSError."""
    archive = _archive(tmp_path, ["A.mid", "B.mid", "C.mid"])
    raw = archive.read_bytes()
    truncated = tmp_path / "truncated.zip"
    # Drop the leading local headers; the central directory at the tail stays
    # intact, so the member list still reads and the offsets go negative.
    truncated.write_bytes(raw[120:])

    with pytest.raises(ValueError) as excinfo:
        _safe_extract_midi(truncated, tmp_path / "out")

    message = str(excinfo.value)
    assert "truncated or corrupt" in message
    assert "re-upload" in message.lower()
    # The operator needs the numbers to confirm the diagnosis themselves.
    assert "compressed bytes" in message


def test_truncation_raises_before_any_file_is_written(tmp_path):
    archive = _archive(tmp_path, ["A.mid", "B.mid", "C.mid"])
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(archive.read_bytes()[120:])
    destination = tmp_path / "out"

    with pytest.raises(ValueError):
        _safe_extract_midi(truncated, destination)

    assert list(destination.glob("*.mid")) == []


def test_non_zip_input_is_reported_clearly(tmp_path):
    """Audio supplied where a MIDI zip was expected, for instance."""
    path = tmp_path / "not.zip"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    with pytest.raises(ValueError, match="not a readable zip archive"):
        _safe_extract_midi(path, tmp_path / "out")


def test_the_committed_archive_is_rejected(tmp_path):
    """The archive stored in this repository is the corrupt one."""
    import base64
    import json

    source = Path("stem_remix_retry_jobs/hypervigilant-bass-pad-remix-20260804-1252.json")
    if not source.exists():  # pragma: no cover - record may be pruned
        pytest.skip("historical job record not present")

    encoded = json.loads(source.read_text(encoding="utf-8"))["input"].get("midi_zip_base64")
    if not encoded:  # pragma: no cover
        pytest.skip("record carries no MIDI archive")

    path = tmp_path / "committed.zip"
    path.write_bytes(base64.b64decode(encoded))

    # It lists cleanly - which is exactly why it looked usable.
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
        assert len(archive.namelist()) == 7

    with pytest.raises(ValueError, match="truncated or corrupt"):
        _safe_extract_midi(path, tmp_path / "out")


def test_unsafe_member_paths_are_still_rejected(tmp_path):
    """The pre-existing traversal guard must survive the new validation."""
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.mid", _midi_bytes())

    extracted = _safe_extract_midi(path, tmp_path / "out")
    # Path().name strips the traversal, so the file lands inside destination.
    assert all((tmp_path / "out").resolve() in p.parents for p in extracted)
