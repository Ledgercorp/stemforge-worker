from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "untitled"


def run_checked(command: list[str], *, timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


def ffprobe(path: Path) -> dict:
    proc = run_checked(["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,bits_per_sample,width,height,r_frame_rate:format=duration,size,format_name,tags", "-of", "json", str(path)])
    return json.loads(proc.stdout)


def ensure_pcm_wav(source: Path, destination: Path, *, sample_rate: int = 48000, channels: int = 2) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vn", "-ar", str(sample_rate), "-ac", str(channels), "-c:a", "pcm_f32le", str(destination)])
    return destination


def load_audio(path: Path, *, always_stereo: bool = True) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if always_stereo:
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
    return audio, int(sample_rate)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_24")
    return path


def db(value: float, floor: float = -180.0) -> float:
    return floor if value <= 0 else 20.0 * math.log10(value)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-20))


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def stereo_correlation(audio: np.ndarray) -> float | None:
    if audio.ndim != 2 or audio.shape[1] < 2 or len(audio) < 2:
        return None
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right)) + 1e-20
    return float(np.dot(left, right) / denominator)


def integrated_loudness(audio: np.ndarray, sample_rate: int) -> float | None:
    try:
        import pyloudnorm as pyln
        return float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except Exception:
        return None


def true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    if audio.size == 0:
        return 0.0
    maximum = 0.0
    for channel in range(audio.shape[1]):
        maximum = max(maximum, float(np.max(np.abs(signal.resample_poly(audio[:, channel], oversample, 1)))))
    return maximum


def normalize_loudness(audio: np.ndarray, sample_rate: int, target_lufs: float, *, maximum_gain_db: float = 12.0) -> tuple[np.ndarray, dict]:
    measured = integrated_loudness(audio, sample_rate)
    if measured is None or not np.isfinite(measured):
        measured = db(rms(audio))
    gain_db = float(np.clip(target_lufs - measured, -maximum_gain_db, maximum_gain_db))
    return audio * (10.0 ** (gain_db / 20.0)), {"measured_lufs": round(float(measured), 3), "target_lufs": float(target_lufs), "gain_db": round(gain_db, 3)}


def linked_compressor(audio: np.ndarray, sample_rate: int, *, threshold_db: float, ratio: float, attack_ms: float = 25.0, release_ms: float = 180.0, knee_db: float = 5.0) -> np.ndarray:
    detector_db = 20.0 * np.log10(np.max(np.abs(audio), axis=1) + 1e-12)
    lower, upper = threshold_db - knee_db / 2.0, threshold_db + knee_db / 2.0
    gain_reduction_db = np.zeros_like(detector_db)
    hard = detector_db > upper
    gain_reduction_db[hard] = threshold_db + (detector_db[hard] - threshold_db) / ratio - detector_db[hard]
    knee = (detector_db >= lower) & (detector_db <= upper)
    x = detector_db[knee] - lower
    gain_reduction_db[knee] = (1.0 / ratio - 1.0) * (x**2) / (2.0 * max(knee_db, 1e-6))
    attack = math.exp(-1.0 / max(1.0, sample_rate * attack_ms / 1000.0))
    release = math.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0))
    smoothed = np.empty_like(gain_reduction_db)
    state = 0.0
    for index, target in enumerate(gain_reduction_db):
        coefficient = attack if target < state else release
        state = coefficient * state + (1.0 - coefficient) * target
        smoothed[index] = state
    return audio * (10.0 ** (smoothed / 20.0))[:, None]


def limit_peak(audio: np.ndarray, ceiling_dbfs: float = -1.0) -> tuple[np.ndarray, dict]:
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    maximum = peak(audio)
    reduction_db = 0.0
    if maximum > ceiling:
        reduction_db = db(ceiling / maximum)
        audio = audio * (ceiling / maximum)
    return audio, {"ceiling_dbfs": float(ceiling_dbfs), "limiter_reduction_db": round(float(reduction_db), 3)}


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 20.0) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def apply_mid_side_width(audio: np.ndarray, width: float) -> np.ndarray:
    if audio.shape[1] < 2:
        return audio
    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5 * float(width)
    return np.column_stack((mid + side, mid - side)).astype(np.float32)
