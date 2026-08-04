from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.actions_v2 import SUPPORTED_ACTIONS
from app.naturalize_dsp_v2 import (
    NOMINAL_COCKTAIL,
    PARAMETER_RANGES,
    ROLE_MULTIPLIERS,
    processing_order,
    render_dsp_role,
    resolve_intensity,
    resolve_parameters,
    resolve_passes,
)
from app.naturalize_v2 import naturalize_arrays, naturalize_audio_job
from app.storage import WORKSPACE
from app.system_v2 import BUILD, VERSION, system_info_job
import app.vocoder_v2 as vocoder_v2


def _parts(sample_rate: int = 48000, seconds: float = 0.65):
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    envelope = np.minimum(1.0, time * 30.0)
    vocal_mono = envelope * (
        0.08 * np.sin(2.0 * np.pi * 220.0 * time)
        + 0.022 * np.sin(2.0 * np.pi * 880.0 * time)
    )
    harmonic_left = (
        0.13 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.032 * np.sin(2.0 * np.pi * 3100.0 * time)
    )
    harmonic_right = (
        0.13 * np.sin(2.0 * np.pi * 110.0 * time + 0.02)
        + 0.032 * np.sin(2.0 * np.pi * 3100.0 * time - 0.02)
    )
    clicks = np.zeros_like(time)
    clicks[:: max(1, sample_rate // 8)] = 0.08
    vocal = np.column_stack((vocal_mono, vocal_mono)).astype(np.float32)
    harmonic = np.column_stack((harmonic_left, harmonic_right)).astype(np.float32)
    percussion = np.column_stack((clicks, clicks)).astype(np.float32)
    source = (vocal + harmonic + percussion).astype(np.float32)
    return source, vocal, harmonic, percussion


def _quiet_parameters() -> dict:
    return {
        "noise_floor_enabled": False,
        "timing_jitter_enabled": False,
        "saturation_enabled": False,
        "multiband_compression": False,
        "denoise_reduction_db": 3.0,
    }


def test_v240_capability_contract() -> None:
    info = system_info_job({})
    assert "naturalize" in SUPPORTED_ACTIONS
    assert VERSION == "2.4.0"
    assert BUILD == "v2.4.0-naturalize-balanced-vocoder"
    assert info["naturalize"]["preset"] == "Naturalize Balanced"
    assert info["naturalize"]["default_intensity"] == 1.0
    assert info["naturalize"]["intensity_range"] == [0.6, 1.4]
    assert info["naturalize"]["maximum_full_passes"] == 2
    assert info["naturalize"]["second_pass_maximum_intensity_scale"] == 0.70
    policy = info["naturalize"]["agent_policy"]
    assert policy["denoise_must_follow_all_modulation"] is True
    assert policy["vocoder_must_follow_denoise"] is True
    assert policy["not_a_detection_evasion_tool"] is True
    assert info["quality_gates"]["vocoder_fallback_to_dsp"] is True


def test_parameter_ranges_defaults_and_clamping() -> None:
    assert NOMINAL_COCKTAIL["primary_vibrato"] == {
        "rate_hz": 15.0,
        "depth_cents": 10.0,
        "waveform": "sine",
    }
    assert NOMINAL_COCKTAIL["secondary_vibrato"]["depth_cents"] == 9.0
    assert NOMINAL_COCKTAIL["flanger"]["modulation_rate_hz"] == 0.9
    assert NOMINAL_COCKTAIL["flanger"]["wet_mix"] == 0.30
    assert PARAMETER_RANGES["denoise_reduction_db"] == [3.0, 8.0]
    assert PARAMETER_RANGES["noise_profile_ms"] == [200.0, 500.0]
    parameters = resolve_parameters(
        {
            "primary_vibrato_rate_hz": 999,
            "primary_vibrato_depth_cents": -3,
            "flanger_feedback": 1.0,
            "flanger_wet_mix": 0.99,
            "denoise_reduction_db": 99,
            "noise_profile_ms": 1,
        }
    )
    assert parameters["primary_vibrato_rate_hz"] == 16.0
    assert parameters["primary_vibrato_depth_cents"] == 8.0
    assert parameters["flanger_feedback"] == 0.15
    assert parameters["flanger_wet_mix"] == 0.40
    assert parameters["denoise_reduction_db"] == 8.0
    assert parameters["noise_profile_ms"] == 200.0


def test_intensity_pass_and_role_bounds() -> None:
    assert resolve_intensity({}) == 1.0
    assert resolve_intensity({"intensity": 0}) == 0.0
    assert resolve_intensity({"intensity": 0.4}) == 0.6
    assert resolve_intensity({"depth": 100}) == 1.0
    assert resolve_intensity({"intensity": 500}) == 1.4
    assert resolve_passes({"passes": 99, "second_pass_intensity": 2}) == (2, 0.70)
    assert 1.1 <= ROLE_MULTIPLIERS["vocal"] <= 1.3
    assert 0.9 <= ROLE_MULTIPLIERS["harmonic"] <= 1.1
    assert 0.5 <= ROLE_MULTIPLIERS["percussion"] <= 0.8


def test_zero_intensity_exact_bypass() -> None:
    source, _, _, _ = _parts()
    output, report = naturalize_arrays(
        source, 48000, mode="quick", intensity=0, vocoder=False
    )
    assert report["status"] == "completed"
    assert np.max(np.abs(output - source)) < 1e-7
    assert report["retain_pre_denoise"] is True
    assert report["operations"][4]["operation"] == "constrained_post_cocktail_denoise"
    assert report["operations"][4]["bypassed"] is True


def test_cocktail_noise_floor_and_denoise_order() -> None:
    source, _, _, _ = _parts(seconds=0.35)
    output, pre_denoise, role_report = render_dsp_role(
        source,
        48000,
        "vocal",
        0.6,
        1,
        0.65,
        resolve_parameters(
            {
                "timing_jitter_enabled": False,
                "saturation_enabled": False,
                "noise_floor_enabled": True,
                "noise_floor_dbfs": -66,
                "denoise_reduction_db": 3,
            }
        ),
        99,
    )
    operations = role_report["operations"]
    names = [item["operation"] for item in operations]
    assert names[:4] == ["vibrato", "tremolo", "vibrato", "flanger"]
    assert names.index("flanger") < names.index("noise_floor_restoration")
    assert names.index("noise_floor_restoration") < names.index(
        "constrained_post_cocktail_denoise"
    )
    denoise = operations[-1]
    assert denoise["order_requirement"] == "post_cocktail_only"
    assert 3 <= denoise["maximum_reduction_db"] <= 8
    assert 200 <= denoise["noise_profile"]["duration_ms"] <= 500
    assert denoise["spectral_uniformity_restoration"] is False
    assert output.shape == source.shape == pre_denoise.shape


def test_second_pass_cap_is_logged() -> None:
    source, _, _, _ = _parts(seconds=0.30)
    _, _, report = render_dsp_role(
        source,
        48000,
        "percussion",
        0.6,
        2,
        0.70,
        resolve_parameters(_quiet_parameters()),
        42,
    )
    assert len(report["passes"]) == 2
    assert report["passes"][0]["scale"] == 1.0
    assert report["passes"][1]["scale"] == 0.70
    assert report["passes"][1]["intensity"] <= 0.42 + 1e-6


def test_damaged_stems_anchor_then_abort_when_still_unsafe() -> None:
    source, vocal, harmonic, percussion = _parts(seconds=0.35)
    _, report = naturalize_arrays(
        source,
        48000,
        mode="surgical",
        intensity=0.6,
        stem_groups=[
            {"name": "Lead Vocals", "role": "vocal", "audio": vocal},
            {"name": "Music", "role": "harmonic", "audio": harmonic * 0.35},
            {"name": "Drums", "role": "percussion", "audio": percussion},
        ],
        parameters=_quiet_parameters(),
        vocoder=False,
    )
    assert report["status"] == "rejected"
    assert report["reconstruction_gate"]["accepted"] is False
    assert report["anchor_decision"]["mix_source"] == "master_anchored_stem_delta"
    assert report["safety"]["accepted"] is False
    assert report["safety"]["reasons"]
    assert {item["role"] for item in report["stem_routing"]} == {
        "vocal",
        "harmonic",
        "percussion",
    }


def test_unavailable_vocoder_falls_back_to_dsp(monkeypatch) -> None:
    source, _, _, _ = _parts(seconds=0.30)
    monkeypatch.setattr(
        vocoder_v2,
        "availability",
        lambda backend: {"available": False, "required_modules": [backend]},
    )
    config = vocoder_v2.resolve_vocoder({"vocoder": "vocos"}, "surgical")
    output, report = vocoder_v2.apply_vocoder(
        source, 48000, config, "vocal", 1.0, 5
    )
    assert np.max(np.abs(output - source)) < 1e-7
    assert report["bypassed"] is True
    assert report["reason"] == "backend_unavailable"


def test_transparent_mock_vocoder_is_accepted(monkeypatch) -> None:
    _, vocal, _, _ = _parts(seconds=0.30)
    monkeypatch.setattr(vocoder_v2, "availability", lambda backend: {"available": True})
    monkeypatch.setitem(
        vocoder_v2.RUNNERS,
        "vocos",
        lambda mono, sr, model, perturbation, seed: (
            mono * 0.995,
            {"backend": "vocos", "model": model, "feature_perturbation": perturbation},
        ),
    )
    config = vocoder_v2.resolve_vocoder({"vocoder": "vocos"}, "surgical")
    output, report = vocoder_v2.apply_vocoder(
        vocal, 48000, config, "vocal", 1.0, 7
    )
    assert output.shape == vocal.shape
    assert report["bypassed"] is False
    assert 0.70 <= report["wet_mix"] <= 1.0
    assert report["safety"]["accepted"] is True


def test_metallic_mock_vocoder_is_rejected(monkeypatch) -> None:
    _, vocal, _, _ = _parts(seconds=0.30)
    monkeypatch.setattr(vocoder_v2, "availability", lambda backend: {"available": True})

    def metallic(mono, sr, model, perturbation, seed):
        del model, perturbation, seed
        alternating = np.where(np.arange(len(mono)) % 2, 0.7, -0.7).astype(np.float32)
        return alternating, {"backend": "vocos", "sample_rate": sr}

    monkeypatch.setitem(vocoder_v2.RUNNERS, "vocos", metallic)
    config = vocoder_v2.resolve_vocoder({"vocoder": "vocos"}, "surgical")
    output, report = vocoder_v2.apply_vocoder(
        vocal, 48000, config, "vocal", 1.0, 11
    )
    assert np.max(np.abs(output - vocal)) < 1e-7
    assert report["bypassed"] is True
    assert report["reason"] == "quality_gate_fallback_to_dsp"
    assert report["safety"]["accepted"] is False


def test_job_retains_original_and_pre_denoise(tmp_path: Path) -> None:
    source, _, _, _ = _parts(seconds=0.25)
    input_root = WORKSPACE / "test_naturalize_v240_quick"
    input_root.mkdir(parents=True, exist_ok=True)
    source_path = input_root / "source.wav"
    sf.write(source_path, source, 48000, subtype="PCM_24")
    report = naturalize_audio_job(
        {
            "action": "naturalize",
            "artist": "test",
            "song": "balanced quick",
            "audio_path": str(source_path),
            "mode": "quick",
            "intensity": 0,
            "vocoder": False,
            "publish_outputs": False,
            "job_dir": str(tmp_path / "job"),
        }
    )
    assert report["status"] == "completed"
    assert report["preset"] == "Naturalize Balanced"
    assert Path(report["processed_path"]).exists()
    assert Path(report["original_reference_path"]).exists()
    assert Path(report["pre_denoise_reference_path"]).exists()


def test_auto_prefers_surgical_when_stems_exist(tmp_path: Path) -> None:
    source, vocal, harmonic, percussion = _parts(seconds=0.25)
    root = WORKSPACE / "test_naturalize_v240_surgical"
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, audio in (("source", source), ("vocals", vocal), ("music", harmonic), ("drums", percussion)):
        path = root / f"{name}.wav"
        sf.write(path, audio, 48000, subtype="PCM_24")
        paths[name] = path
    report = naturalize_audio_job(
        {
            "action": "naturalize",
            "artist": "test",
            "song": "auto surgical",
            "audio_path": str(paths["source"]),
            "mode": "auto",
            "intensity": 0,
            "vocoder": False,
            "publish_outputs": False,
            "job_dir": str(tmp_path / "surgical_job"),
            "stems": [
                {"name": "Lead Vocals", "role": "vocal", "path": str(paths["vocals"])},
                {"name": "Guitars and Bass", "role": "harmonic", "path": str(paths["music"])},
                {"name": "Drums", "role": "percussion", "path": str(paths["drums"])},
            ],
        }
    )
    assert report["status"] == "completed"
    assert report["resolved_mode"] == "surgical"
    assert report["automatic_mode_selection"] is True
    assert report["fallback_reason"] is None
    roles = {item["role"] for item in report["agent_log"]["parameters"]["stem_routing"]}
    assert roles == {"vocal", "harmonic", "percussion"}


def test_vocoder_is_last_in_priority_order() -> None:
    order = processing_order()
    assert order[-2] == "constrained_adaptive_denoise_after_modulation"
    assert order[-1] == "optional_neural_vocoder_after_denoise"
