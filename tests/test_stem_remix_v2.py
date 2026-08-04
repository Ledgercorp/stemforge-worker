from __future__ import annotations

import numpy as np

from app.stem_remix_v2 import (
    _build_premaster,
    _reconstruction_diagnostics,
    _select_mode,
)


def _stereo(value: np.ndarray) -> np.ndarray:
    return np.column_stack((value, value)).astype(np.float32)


def test_exact_stems_are_eligible_for_full_mix() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    stem_a = _stereo(0.15 * np.sin(2.0 * np.pi * 110.0 * time))
    stem_b = _stereo(0.08 * np.sin(2.0 * np.pi * 440.0 * time))
    master = stem_a + stem_b

    diagnostics = _reconstruction_diagnostics([stem_a, stem_b], master)
    decision = _select_mode({"remix_mode": "auto"}, diagnostics)

    assert diagnostics["safe_for_full_stem_mix"] is True
    assert diagnostics["null_residual_relative_db"] < -100.0
    assert decision["selected"] == "full_stem_mix"


def test_imperfect_stems_fall_back_to_master_anchored_delta() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    master = _stereo(0.18 * np.sin(2.0 * np.pi * 220.0 * time))
    unrelated = _stereo(0.08 * np.sin(2.0 * np.pi * 3100.0 * time))

    diagnostics = _reconstruction_diagnostics([unrelated], master)
    decision = _select_mode({"remix_mode": "auto"}, diagnostics)

    assert diagnostics["safe_for_full_stem_mix"] is False
    assert decision["selected"] == "master_anchored_delta"


def test_forced_unsafe_full_mix_is_rejected() -> None:
    diagnostics = {
        "safe_for_full_stem_mix": False,
        "least_squares_gain": 1.0,
    }
    decision = _select_mode({"remix_mode": "full_stem_mix"}, diagnostics)
    assert decision["status"] == "rejected"
    assert decision["selected"] is None


def test_auxiliary_pad_changes_only_its_active_region() -> None:
    sample_rate = 48000
    length = sample_rate * 4
    master = np.zeros((length, 2), dtype=np.float32)
    pad = np.zeros_like(master)
    start = sample_rate
    end = sample_rate * 3
    pad[start:end] = 0.05

    prepared = [
        {
            "name": "Breakdown Bass Pad",
            "role": "bass_pad",
            "overlay": True,
            "gain_db": 0.0,
            "delta_amount": 1.0,
            "raw": pad,
            "processed": pad,
        }
    ]

    result, report = _build_premaster(
        prepared,
        master,
        "balanced",
        "master_anchored_delta",
        1.0,
    )

    assert np.max(np.abs(result[:start] - master[:start])) == 0.0
    assert np.max(np.abs(result[end:] - master[end:])) == 0.0
    assert np.max(np.abs(result[start:end])) > 0.0
    assert report[0]["mode"] == "direct_overlay"


def test_delta_mix_retains_master_when_processing_is_identity() -> None:
    sample_rate = 48000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    stem = _stereo(0.05 * np.sin(2.0 * np.pi * 330.0 * time))
    master = stem.copy()
    prepared = [
        {
            "name": "Guitar",
            "role": "guitar",
            "overlay": False,
            "gain_db": 0.0,
            "delta_amount": 1.0,
            "raw": stem,
            "processed": stem,
        }
    ]

    result, _ = _build_premaster(
        prepared,
        master,
        "balanced",
        "master_anchored_delta",
        1.0,
    )

    # The balanced profile intentionally applies a -0.2 dB guitar fader move,
    # but the coherent master remains the foundation and the change is small.
    correlation = np.corrcoef(master[:, 0], result[:, 0])[0, 1]
    assert correlation > 0.999
    assert np.max(np.abs(result - master)) < 0.01
