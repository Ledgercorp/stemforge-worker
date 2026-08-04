from __future__ import annotations

from pathlib import Path

import app.full_pass_v2 as full_pass_v2


def _install_stubs(monkeypatch, tmp_path: Path, captured: dict) -> None:
    master = tmp_path / "master.wav"
    master.write_bytes(b"placeholder")
    stem = tmp_path / "vocals.wav"
    stem.write_bytes(b"placeholder")
    processed = tmp_path / "processed.wav"
    processed.write_bytes(b"placeholder")

    def materialize(payload, prefix, destination, required=True):
        del payload, destination, required
        return master if prefix == "master" else None

    monkeypatch.setattr(full_pass_v2, "materialize_input", materialize)
    monkeypatch.setattr(
        full_pass_v2,
        "_materialize_stems",
        lambda specs, job_root: [("Lead Vocals", stem)],
    )
    monkeypatch.setattr(
        full_pass_v2,
        "inspect_stems_v2_job",
        lambda payload: {
            "status": "completed",
            "stem_count": 1,
            "stems": [],
            "reconstruction": {"safe_for_stem_remix": True},
        },
    )

    def naturalize(payload):
        captured.update(payload)
        return {
            "status": "completed",
            "processed_path": str(processed),
            "preset": "Naturalize Balanced",
            "requested_intensity": payload["intensity"],
            "intensity": payload["intensity"],
            "intensity_was_reduced": False,
            "passes": {"requested": payload.get("passes", 1)},
            "parameters": payload.get("parameters"),
            "attempts": [],
            "stem_routing": [],
            "vocoder": {"request": payload.get("vocoder")},
            "safety": {"accepted": True},
        }

    monkeypatch.setattr(full_pass_v2, "naturalize_audio_job", naturalize)
    monkeypatch.setattr(
        full_pass_v2,
        "master_audio_job",
        lambda payload: {
            "status": "completed",
            "source_metrics": {},
            "candidates": [],
            "report_output": None,
        },
    )
    monkeypatch.setattr(
        full_pass_v2,
        "publish_file",
        lambda *args, **kwargs: {"storage_mode": "test"},
    )


def test_full_pass_true_uses_v240_balanced_default(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    _install_stubs(monkeypatch, tmp_path, captured)
    report = full_pass_v2.full_pass_job(
        {
            "artist": "test",
            "song": "default bridge",
            "master_path": str(tmp_path / "master.wav"),
            "stems": [{"name": "Lead Vocals", "path": str(tmp_path / "vocals.wav")}],
            "naturalize": True,
        }
    )
    assert report["status"] == "completed"
    assert captured["intensity"] == 1.0
    assert captured["mode"] == "auto"
    assert captured["retain_original"] is True
    assert captured["retain_pre_denoise"] is True
    assert captured["publish_outputs"] is False
    assert report["engine"] == "StemForge full-pass engine v2.2"
    assert report["naturalize"]["preset"] == "Naturalize Balanced"


def test_full_pass_forwards_complete_naturalize_configuration(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    _install_stubs(monkeypatch, tmp_path, captured)
    naturalize = {
        "mode": "surgical",
        "intensity": 1.2,
        "passes": 2,
        "second_pass_intensity": 0.65,
        "vocoder": {
            "backend": "bigvgan",
            "feature_perturbation": 0.001,
        },
        "parameters": {
            "denoise_reduction_db": 4.5,
            "noise_floor_dbfs": -68,
        },
        "retain_pre_denoise": True,
        "separate_if_needed": True,
    }
    report = full_pass_v2.full_pass_job(
        {
            "artist": "test",
            "song": "advanced bridge",
            "master_path": str(tmp_path / "master.wav"),
            "stems": [{"name": "Lead Vocals", "path": str(tmp_path / "vocals.wav")}],
            "naturalize": naturalize,
        }
    )
    assert report["status"] == "completed"
    assert captured["mode"] == "surgical"
    assert captured["intensity"] == 1.2
    assert captured["passes"] == 2
    assert captured["second_pass_intensity"] == 0.65
    assert captured["vocoder"]["backend"] == "bigvgan"
    assert captured["parameters"]["denoise_reduction_db"] == 4.5
    assert captured["parameters"]["noise_floor_dbfs"] == -68
    assert captured["separate_if_needed"] is True
    assert report["naturalize"]["vocoder"]["request"]["backend"] == "bigvgan"
