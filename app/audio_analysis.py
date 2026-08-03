from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from app.utils import download_file, temporary_job_dir


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels,bits_per_sample,codec_name:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _spectral_summary(path: Path) -> dict:
    with sf.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        block = audio.read(
            min(audio.frames, sample_rate * 120),
            dtype="float32",
            always_2d=True,
        )

    mono = block.mean(axis=1)
    freqs, psd = signal.welch(
        mono,
        sample_rate,
        nperseg=min(8192, max(1024, len(mono))),
    )

    total = float(np.trapezoid(psd, freqs) + 1e-18)
    bands = {}

    for name, low, high in [
        ("sub", 20, 80),
        ("bass", 80, 250),
        ("low_mid", 250, 800),
        ("mid", 800, 2500),
        ("presence", 2500, 5000),
        ("high", 5000, 12000),
        ("air", 12000, 20000),
    ]:
        mask = (freqs >= low) & (freqs < high)
        value = float(np.trapezoid(psd[mask], freqs[mask]) / total) if np.any(mask) else 0.0
        bands[name] = round(value, 7)

    peak = float(np.max(np.abs(block)))
    rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64) + 1e-18))

    return {
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "crest_factor_db": round(
            20 * np.log10(max(peak, 1e-12))
            - 20 * np.log10(max(rms, 1e-12)),
            3,
        ),
        "spectral_energy_distribution": bands,
    }


def analyze_audio_url(payload: dict) -> dict:
    audio_url = payload.get("audio_url")
    if not audio_url:
        return {"status": "rejected", "reason": "audio_url is required."}

    with temporary_job_dir("analysis") as temp_dir:
        path = Path(temp_dir) / "audio"
        download_file(audio_url, path)

        return {
            "status": "completed",
            "probe": _probe(path),
            "analysis": _spectral_summary(path),
        }
