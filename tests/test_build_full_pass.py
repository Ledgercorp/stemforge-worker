"""Tests for building a full_pass request from an ingest record.

The record is the only link between "these files are stored" and "render
them", so a mistake here points a long, expensive job at the wrong object -
or at no object. Classification is checked against the real filenames both
songs produced.
"""

from __future__ import annotations

import json

import pytest

from tools import build_full_pass, validate_jobs


def _record(filename: str, key: str | None = None) -> dict:
    return {
        "filename": filename,
        "storage_key": key or f"incoming/sounddecay/abc123/{filename}",
        "size_bytes": 1024,
    }


HYPERVIGILANT = [
    _record("Hypervigilant_Master.wav"),
    _record("Lead_Vocals.wav"),
    _record("Backing_Vocals.wav"),
    _record("Other.wav"),
    _record("Percussion.wav"),
    _record("Drums.wav"),
    _record("Synth.wav"),
    _record("Bass.wav"),
    _record("Guitar.wav"),
]


def test_master_stems_and_midi_are_separated():
    records = HYPERVIGILANT + [_record("sweet-sixteen-midi.zip")]
    master, midi, stems = build_full_pass.classify(records)

    assert master["filename"] == "Hypervigilant_Master.wav"
    assert midi["filename"] == "sweet-sixteen-midi.zip"
    assert len(stems) == 8
    # The master must not also appear as a stem.
    assert all("Master" not in s["filename"] for s in stems)


def test_job_references_keys_never_urls():
    job, _ = build_full_pass.build(HYPERVIGILANT, song="Hypervigilant")
    serialized = json.dumps(job)

    assert "storage_key" in serialized
    assert "drive.google.com" not in serialized
    assert job["input"]["master_storage_key"].endswith("Hypervigilant_Master.wav")
    assert all("storage_key" in stem for stem in job["input"]["stems"])


def test_underscored_filenames_become_readable_stem_names():
    job, _ = build_full_pass.build(HYPERVIGILANT, song="Hypervigilant")
    names = {stem["name"] for stem in job["input"]["stems"]}
    assert "Lead Vocals" in names
    assert "Backing Vocals" in names


def test_routing_is_reported_for_every_stem():
    """A stem that routes to the wrong role is processed at the wrong strength."""
    _, routing = build_full_pass.build(HYPERVIGILANT, song="Hypervigilant")
    roles = dict(routing)

    assert roles["Lead Vocals"] == "vocal"
    assert roles["Backing Vocals"] == "vocal"
    assert roles["Drums"] == "percussion"
    assert roles["Percussion"] == "percussion"
    assert roles["Bass"] == roles["Guitar"] == roles["Synth"] == "harmonic"


def test_a_name_with_no_keyword_is_flagged_not_hidden():
    """Sweet Sixteen has an `FX` track, which matches no role vocabulary."""
    records = [_record("Sweet_Sixteen_Master.wav"), _record("FX.wav")]
    _, routing = build_full_pass.build(records, song="Sweet Sixteen")
    assert dict(routing)["FX"] == "other"


def test_parenthesised_names_still_route():
    """The Sweet Sixteen archive names its members `Sixteen  (Drums)`."""
    records = [
        _record("Sweet_Sixteen_Master.wav"),
        _record("Sixteen  (Drums).wav"),
        _record("Sixteen  (Vocals).wav"),
    ]
    _, routing = build_full_pass.build(records, song="Sweet Sixteen")
    roles = dict(routing)
    assert roles["Sixteen  (Drums)"] == "percussion"
    assert roles["Sixteen  (Vocals)"] == "vocal"


def test_a_missing_master_is_refused():
    with pytest.raises(build_full_pass.BuildError, match="no master"):
        build_full_pass.build([_record("Drums.wav")], song="X")


def test_a_second_master_is_refused():
    """Silently picking one of two would render against the wrong source."""
    records = [
        _record("Hypervigilant_Master.wav"),
        _record("Hypervigilant_Master_v2.wav"),
        _record("Drums.wav"),
    ]
    with pytest.raises(build_full_pass.BuildError, match="two files look like"):
        build_full_pass.build(records, song="X")


def test_a_master_with_no_stems_is_refused():
    with pytest.raises(build_full_pass.BuildError, match="no stems"):
        build_full_pass.build([_record("Hypervigilant_Master.wav")], song="X")


def test_midi_is_optional():
    job, _ = build_full_pass.build(HYPERVIGILANT, song="Hypervigilant")
    assert "midi_zip_storage_key" not in job["input"]


def test_quality_settings_match_the_delivered_pass():
    job, _ = build_full_pass.build(HYPERVIGILANT, song="Hypervigilant")
    naturalize = job["input"]["naturalize"]

    assert naturalize["intensity"] == 1.4
    assert naturalize["mode"] == "auto"
    assert job["input"]["profiles"] == ["dynamic", "balanced", "dense"]
    assert job["input"]["required_build"] == "v2.4.0-naturalize-balanced-vocoder"
    # The worker must have longer to finish than the client waits to time out.
    assert job["policy"]["executionTimeout"] <= job["policy"]["ttl"]


def test_the_built_job_satisfies_the_relay_contract(tmp_path):
    """The point of the tool: what it emits must be runnable as written."""
    records = HYPERVIGILANT + [_record("sweet-sixteen-midi.zip")]
    job, _ = build_full_pass.build(records, song="Sweet Sixteen")

    path = tmp_path / "full-pass-sweet-sixteen-20260805-1600.json"
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    errors = [i for i in validate_jobs.validate_file(path) if i.level == validate_jobs.ERROR]
    assert errors == []


def test_cli_writes_a_job_from_a_record_file(tmp_path, capsys):
    record = tmp_path / "hypervigilant.json"
    record.write_text(json.dumps({"ingested": HYPERVIGILANT}), encoding="utf-8")
    out = tmp_path / "job.json"

    assert build_full_pass.main(
        [str(record), "--song", "Hypervigilant", "--out", str(out)]
    ) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["input"]["song"] == "Hypervigilant"
    assert len(written["input"]["stems"]) == 8


def test_cli_reports_a_bad_record_without_writing(tmp_path):
    record = tmp_path / "broken.json"
    record.write_text(json.dumps({"ingested": [_record("Drums.wav")]}), encoding="utf-8")
    out = tmp_path / "job.json"

    assert build_full_pass.main(
        [str(record), "--song", "X", "--out", str(out)]
    ) == 1
    assert not out.exists()
