from __future__ import annotations

import numpy as np

from app.quality_v2 import compare_audio_quality, quality_gate


def _tone(sample_rate: int = 48000, seconds: float = 4.0) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    mono = 0.22 * np.sin(2.0 * np.pi * 440.0 * time)
    return np.column_stack((mono, mono)).astype(np.float32)


def _metrics(true_peak: float = -2.0) -> dict:
    return {"clipped_sample_count": 0, "true_peak_dbtp": true_peak}


def test_clean_gain_change_passes_quality_gate() -> None:
    reference = _tone()
    candidate = reference * 0.92
    result = quality_gate(
        reference,
        candidate,
        48000,
        _metrics(),
        mode="master_anchored_delta",
    )
    assert result["accepted"] is True
    assert result["reasons"] == []
    assert result["comparison"]["waveform_correlation"] > 0.999


def test_static_like_alternating_noise_is_rejected() -> None:
    reference = _tone()
    alternating = np.where(np.arange(len(reference)) % 2 == 0, 1.0, -1.0)
    noise = np.column_stack((alternating, -alternating)).astype(np.float32) * 0.055
    candidate = reference + noise
    result = quality_gate(
        reference,
        candidate,
        48000,
        _metrics(),
        mode="master_anchored_delta",
    )
    assert result["accepted"] is False
    assert {
        "excess_derivative_energy",
        "transient_or_click_growth",
        "high_frequency_noise_growth",
    }.intersection(result["reasons"])


def test_non_finite_samples_are_rejected() -> None:
    reference = _tone()
    candidate = reference.copy()
    candidate[100, 0] = np.nan
    result = quality_gate(
        reference,
        candidate,
        48000,
        _metrics(),
        mode="master_anchored_delta",
    )
    assert result["accepted"] is False
    assert "non_finite_samples" in result["reasons"]


def test_quality_comparison_reports_level_matched_residual() -> None:
    reference = _tone()
    candidate = reference * 0.5
    comparison = compare_audio_quality(reference, candidate, 48000)
    assert comparison["level_matched_residual_relative_db"] < -100.0
    assert abs(comparison["level_match_gain_db"] - 6.0206) < 0.02
