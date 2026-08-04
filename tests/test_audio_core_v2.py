from __future__ import annotations

import numpy as np
import pytest

from app.audio_core import write_audio


def test_strict_write_refuses_hidden_clipping(tmp_path) -> None:
    audio = np.zeros((4800, 2), dtype=np.float32)
    audio[100, 0] = 1.2
    with pytest.raises(ValueError, match="Refusing to silently clip"):
        write_audio(tmp_path / "overloaded.wav", audio, 48000, allow_clip=False)


def test_write_refuses_non_finite_audio(tmp_path) -> None:
    audio = np.zeros((4800, 2), dtype=np.float32)
    audio[100, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        write_audio(tmp_path / "invalid.wav", audio, 48000, allow_clip=False)


def test_strict_write_preserves_safe_audio(tmp_path) -> None:
    time = np.arange(4800, dtype=np.float64) / 48000.0
    mono = 0.25 * np.sin(2.0 * np.pi * 440.0 * time)
    audio = np.column_stack((mono, mono)).astype(np.float32)
    destination = tmp_path / "safe.wav"
    write_audio(destination, audio, 48000, allow_clip=False)
    assert destination.exists()
    assert destination.stat().st_size > 44
