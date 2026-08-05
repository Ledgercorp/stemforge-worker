"""Regression tests for the Naturalize pumping detector.

The pumping index is the standard deviation of the candidate/reference gain
ratio. It was computed over every frame with the divisor floored at 1e-8,
which is roughly -160 dBFS. Naturalize adds a shaped noise floor between -72
and -60 dBFS as a feature, so across a silent passage the reference approaches
zero while the candidate does not and the ratio diverges. A transparent render
scored in the tens of thousands against a 0.20 threshold, and reducing
intensity could never help because the noise floor is added regardless.

These tests pin both halves of the contract: the detector must ignore silent
frames, and it must still catch real pumping. The threshold itself is
unchanged at 0.20.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.naturalize_dsp_v2 import _envelope_metrics, artifact_assessment

SAMPLE_RATE = 48000
PUMPING_THRESHOLD = 0.20


def _music(seconds: float = 6.0, silence: tuple[float, float] | None = (2.5, 3.0)):
    """A tone with a genuine silent gap, as most real tracks contain."""
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    mono = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    if silence is not None:
        start, end = silence
        mono[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = 0.0
    return np.stack([mono, mono], axis=1).astype(np.float32)


def _with_noise_floor(audio: np.ndarray, dbfs: float = -66.0, seed: int = 5):
    """Add the shaped noise floor Naturalize introduces by design."""
    rng = np.random.default_rng(seed)
    amplitude = 10 ** (dbfs / 20)
    return (audio + rng.normal(0, amplitude, audio.shape)).astype(np.float32)


def test_added_noise_floor_over_silence_is_not_pumping():
    """The exact defect: a transparent render scored 33,394."""
    reference = _music()
    candidate = _with_noise_floor(reference)

    metrics = _envelope_metrics(reference, candidate, SAMPLE_RATE)

    # Sanity: this render really is transparent.
    assert metrics["correlation"] > 0.99
    assert metrics["transient_retention"] > 0.95
    assert metrics["pumping_index"] < PUMPING_THRESHOLD


def test_silent_passage_does_not_dominate_the_statistic():
    """A longer silence must not change the verdict."""
    quiet = _with_noise_floor(_music(silence=(1.0, 4.0)))
    brief = _with_noise_floor(_music(silence=(2.9, 3.0)))

    long_gap = _envelope_metrics(_music(silence=(1.0, 4.0)), quiet, SAMPLE_RATE)
    short_gap = _envelope_metrics(_music(silence=(2.9, 3.0)), brief, SAMPLE_RATE)

    assert long_gap["pumping_index"] < PUMPING_THRESHOLD
    assert short_gap["pumping_index"] < PUMPING_THRESHOLD


def test_genuine_pumping_is_still_detected():
    """The gate must not have been weakened: real pumping still fires."""
    reference = _music(silence=None)
    t = np.arange(len(reference)) / SAMPLE_RATE
    # A compressor breathing hard: 3 Hz gain oscillation, +/- 50%.
    envelope = (1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)).astype(np.float32)
    candidate = (reference * envelope[:, None]).astype(np.float32)

    metrics = _envelope_metrics(reference, candidate, SAMPLE_RATE)
    assert metrics["pumping_index"] > PUMPING_THRESHOLD

    verdict = artifact_assessment(reference, candidate, SAMPLE_RATE, 1.0)
    assert "pumping" in verdict["reasons"]
    assert verdict["accepted"] is False


def test_pumping_is_detected_even_when_the_track_has_silence():
    """Real pumping in a track with gaps must still be caught."""
    reference = _music()
    t = np.arange(len(reference)) / SAMPLE_RATE
    envelope = (1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * t)).astype(np.float32)
    candidate = _with_noise_floor((reference * envelope[:, None]).astype(np.float32))

    metrics = _envelope_metrics(reference, candidate, SAMPLE_RATE)
    assert metrics["pumping_index"] > PUMPING_THRESHOLD


def test_transparent_render_passes_the_whole_gate():
    """End to end: a transparent render is accepted, not merely un-flagged."""
    reference = _music()
    candidate = _with_noise_floor(reference)

    verdict = artifact_assessment(reference, candidate, SAMPLE_RATE, 1.0)
    assert "pumping" not in verdict["reasons"]
    assert "pumping_or_envelope_damage" not in verdict["reasons"]


def test_fully_silent_input_does_not_crash_or_flag():
    silent = np.zeros((SAMPLE_RATE * 2, 2), dtype=np.float32)
    metrics = _envelope_metrics(silent, _with_noise_floor(silent), SAMPLE_RATE)
    assert np.isfinite(metrics["pumping_index"])
    assert metrics["pumping_index"] < PUMPING_THRESHOLD


def test_index_stays_on_a_normalized_scale():
    """A ratio statistic belongs near 0-1; the defect produced 5-6 figures."""
    reference = _music()
    for dbfs in (-72.0, -66.0, -60.0):
        metrics = _envelope_metrics(
            reference, _with_noise_floor(reference, dbfs=dbfs), SAMPLE_RATE
        )
        assert 0.0 <= metrics["pumping_index"] < 1.0, dbfs


@pytest.mark.parametrize("threshold_source", ["app/naturalize_dsp_v2.py"])
def test_threshold_is_unchanged(threshold_source):
    """The fix corrects a measurement; it must not relax the gate."""
    from pathlib import Path

    source = Path(threshold_source).read_text(encoding="utf-8")
    assert 'if float(envelope["pumping_index"]) > 0.20: reasons.append("pumping")' in source
