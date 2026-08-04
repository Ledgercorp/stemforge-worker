from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.full_pass_v2 import _naturalize_config
from app.naturalize_v2 import (
    _resolve_intensity,
    naturalize_arrays,
    naturalize_audio_job,
)
from app.storage import WORKSPACE


def _parts(sample_rate: int = 48000, seconds: float = 1.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    vocal_mono = 0.09 * np.sin(2.0 * np.pi * 220.0 * time)
    vocal_mono += 0.025 * np.sin(2.0 * np.pi * 880.0 * time)
    music_left = 0.14 * np.sin(2.0 * np.pi * 110.0 * time)
    music_left += 0.035 * np.sin(2.0 * np.pi * 3100.0 * time)
    music_right = 0.14 * np.sin(2.0 * np.pi * 110.0 * time + 0.02)
    music_right += 0.035 * np.sin(2.0 * np.pi * 3100.0 * time - 0.02)
    vocal = np.column_stack((vocal_mono, vocal_mono)).astype(np.float32)
    instrumental = np.column_stack((music_left, music_right)).astype(np.float32)
    return (vocal + instrumental).astype(np.float32), vocal, instrumental


def test_intensity_accepts_fraction_or_percent() -> None:
    assert _resolve_intensity({"intensity": 0.4}) == 0.4
    assert _resolve_intensity({"depth": 40}) == 0.4
    assert _resolve_intensity({"intensity": 500}) == 1.0


def test_full_pass_accepts_boolean_string_and_object_naturalize_config() -> None:
    assert _naturalize_config(False) is None
    assert _naturalize_config(True) == {}
    assert _naturalize_config("surgical") == {"mode": "surgical"}
    assert _naturalize_config({"mode": "quick", "intensity": 0.2}) == {
        "mode": "quick",
        "intensity": 0.2,
    }


def test_zero_intensity_is_a_true_non_destructive_bypass() -> None:
    source, _, _ = _parts()
    output, report = naturalize_arrays(source, 48000, mode="quick", intensity=0.0)
    assert report["status"] == "completed", report["safety"]
    assert report["non_destructive"] is True
    assert np.max(np.abs(output - source)) < 1e-6
    assert report["operations"][4]["operation"] == "light_post_modulation_denoise"
    assert report["operations"][4]["bypassed"] is True


def test_quick_naturalize_uses_the_mandatory_order() -> None:
    source, _, _ = _parts()
    output, report = naturalize_arrays(source, 48000, mode="quick", intensity=0.12)
    assert report["status"] == "completed", report["safety"]
    assert output.shape == source.shape
    assert np.all(np.isfinite(output))
    assert report["processing_order"] == [
        "vibrato_15hz_depth_10",
        "tremolo_15hz_depth_10",
        "vibrato_10hz_depth_10",
        "flanger_1ms_modulation_0.90_depth_9",
        "light_denoise_after_modulation",
    ]
    assert [item["operation"] for item in report["operations"][:5]] == [
        "vibrato",
        "tremolo",
        "vibrato",
        "flanger",
        "light_post_modulation_denoise",
    ]
    assert report["honesty_note"].endswith("not a detection tool.")


def test_surgical_mode_anchors_to_master_when_stems_do_not_reconstruct() -> None:
    source, vocal, instrumental = _parts()
    damaged_instrumental = instrumental * 0.4
    output, report = naturalize_arrays(
        source,
        48000,
        mode="surgical",
        intensity=0.1,
        vocal=vocal,
        instrumental=damaged_instrumental,
    )
    assert report["status"] == "completed", report["safety"]
    assert output.shape == source.shape
    assert report["reconstruction_gate"]["accepted"] is False
    assert report["anchor_decision"]["instrumental_source"] == "master_minus_vocal"
    assert report["operations"][0]["operation"] == "surgical_vocal_naturalize"
    assert report["operations"][1]["operation"] == "surgical_instrumental_naturalize"
    assert report["operations"][2]["operation"] == "light_post_modulation_denoise"


def test_auto_job_prefers_surgical_when_vocal_stem_is_available(tmp_path: Path) -> None:
    source, vocal, instrumental = _parts(seconds=0.5)
    input_root = WORKSPACE / "test_naturalize_surgical"
    input_root.mkdir(parents=True, exist_ok=True)
    source_path = input_root / "source.wav"
    vocal_path = input_root / "vocals.wav"
    instrumental_path = input_root / "instrumental.wav"
    sf.write(source_path, source, 48000, subtype="PCM_24")
    sf.write(vocal_path, vocal, 48000, subtype="PCM_24")
    sf.write(instrumental_path, instrumental, 48000, subtype="PCM_24")
    report = naturalize_audio_job(
        {
            "action": "naturalize",
            "artist": "test",
            "song": "auto surgical",
            "audio_path": str(source_path),
            "mode": "auto",
            "intensity": 0.0,
            "publish_outputs": False,
            "job_dir": str(tmp_path / "surgical_job"),
            "stems": [
                {"name": "Lead Vocals", "path": str(vocal_path)},
                {"name": "Instrumental", "path": str(instrumental_path)},
            ],
        }
    )
    assert report["status"] == "completed", report.get("safety")
    assert report["resolved_mode"] == "surgical"
    assert report["automatic_mode_selection"] is True
    assert report["fallback_reason"] is None
    assert report["agent_log"]["preferred_mode_when_stems_available"] == "surgical"


def test_auto_job_falls_back_to_quick_and_retains_ab_reference(tmp_path: Path) -> None:
    source, _, _ = _parts(seconds=0.5)
    input_root = WORKSPACE / "test_naturalize"
    input_root.mkdir(parents=True, exist_ok=True)
    source_path = input_root / "source.wav"
    sf.write(source_path, source, 48000, subtype="PCM_24")
    report = naturalize_audio_job(
        {
            "action": "naturalize",
            "artist": "test",
            "song": "auto quick",
            "audio_path": str(source_path),
            "mode": "auto",
            "intensity": 0.0,
            "publish_outputs": False,
            "job_dir": str(tmp_path / "job"),
        }
    )
    assert report["status"] == "completed", report.get("safety")
    assert report["resolved_mode"] == "quick"
    assert report["automatic_mode_selection"] is True
    assert report["fallback_reason"] is not None
    assert Path(report["processed_path"]).exists()
    assert Path(report["original_reference_path"]).exists()
    assert report["agent_log"]["callable_operation"] == "naturalize"
    assert report["agent_log"]["fallback_without_stems"] == "quick"
