from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy import signal

from app.analysis_v2 import analyze_array
from app.audio_core import db, linked_compressor, peak, rms
from app.quality_v2 import compare_audio_quality

NATURALIZE_DSP_VERSION = "2.0"
NATURALIZE_LABEL = "Naturalness / Authenticity enhancement"
NATURALIZE_PRESET = "Naturalize Balanced"

PARAMETER_RANGES = {
    "primary_vibrato": {"rate_hz": [14.0, 16.0], "depth_cents": [8.0, 12.0]},
    "tremolo": {"rate_hz": [12.0, 16.0], "depth_percent": [6.0, 12.0]},
    "secondary_vibrato": {"rate_hz": [8.0, 11.0], "depth_cents": [6.0, 10.0]},
    "flanger": {
        "delay_ms": [0.8, 1.2],
        "modulation_rate_hz": [0.6, 1.1],
        "depth": [7.0, 10.0],
        "feedback": [0.0, 0.15],
        "wet_mix": [0.20, 0.40],
    },
    "denoise_reduction_db": [3.0, 8.0],
    "noise_profile_ms": [200.0, 500.0],
    "noise_floor_dbfs": [-72.0, -60.0],
    "timing_jitter_ms": [1.5, 3.0],
    "global_intensity": [0.6, 1.4],
}

NOMINAL_COCKTAIL = {
    "primary_vibrato": {"rate_hz": 15.0, "depth_cents": 10.0, "waveform": "sine"},
    "tremolo": {"rate_hz": 15.0, "depth_percent": 10.0, "waveform": "sine"},
    "secondary_vibrato": {"rate_hz": 10.0, "depth_cents": 9.0, "waveform": "sine"},
    "flanger": {
        "delay_ms": 1.0,
        "modulation_rate_hz": 0.9,
        "depth": 9.0,
        "feedback": 0.10,
        "wet_mix": 0.30,
    },
    "denoise": {
        "type": "adaptive_music_spectral",
        "maximum_reduction_db": 5.0,
        "quiet_profile_ms": 350.0,
        "position": "post_cocktail_only",
    },
}

ROLE_MULTIPLIERS = {"vocal": 1.20, "harmonic": 1.00, "percussion": 0.65, "other": 0.90}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "untitled"


def clamp(value: Any, low: float, high: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not np.isfinite(number):
        number = default
    return float(np.clip(number, low, high))


def resolve_intensity(payload: dict) -> float:
    value = float(payload.get("intensity", payload.get("depth", 1.0)))
    if 2.0 < value <= 140.0:
        value /= 100.0
    if not np.isfinite(value):
        raise ValueError("Naturalize intensity must be finite.")
    return 0.0 if value == 0.0 else float(np.clip(value, 0.6, 1.4))


def resolve_passes(payload: dict) -> tuple[int, float]:
    passes = max(1, min(int(payload.get("passes", 1)), 2))
    second = clamp(
        payload.get("second_pass_intensity", payload.get("second_pass_scale", 0.65)),
        0.0,
        0.70,
        0.65,
    )
    return passes, second


def resolve_parameters(payload: dict) -> dict:
    nested = payload.get("parameters")
    nested = dict(nested) if isinstance(nested, dict) else {}

    def read(name: str, default: Any) -> Any:
        return payload.get(name, nested.get(name, default))

    waveform = str(read("tremolo_waveform", "sine")).lower()
    if waveform not in {"sine", "soft_triangle"}:
        waveform = "sine"
    shape = str(read("noise_floor_shape", "pink")).lower()
    if shape not in {"pink", "tape_hiss"}:
        shape = "pink"
    return {
        "primary_vibrato_rate_hz": clamp(read("primary_vibrato_rate_hz", 15), 14, 16, 15),
        "primary_vibrato_depth_cents": clamp(read("primary_vibrato_depth_cents", 10), 8, 12, 10),
        "tremolo_rate_hz": clamp(read("tremolo_rate_hz", 15), 12, 16, 15),
        "tremolo_depth_percent": clamp(read("tremolo_depth_percent", 10), 6, 12, 10),
        "tremolo_waveform": waveform,
        "secondary_vibrato_rate_hz": clamp(read("secondary_vibrato_rate_hz", 10), 8, 11, 10),
        "secondary_vibrato_depth_cents": clamp(read("secondary_vibrato_depth_cents", 9), 6, 10, 9),
        "flanger_delay_ms": clamp(read("flanger_delay_ms", 1), 0.8, 1.2, 1),
        "flanger_modulation_rate_hz": clamp(read("flanger_modulation_rate_hz", 0.9), 0.6, 1.1, 0.9),
        "flanger_depth": clamp(read("flanger_depth", 9), 7, 10, 9),
        "flanger_feedback": clamp(read("flanger_feedback", 0.10), 0, 0.15, 0.10),
        "flanger_wet_mix": clamp(read("flanger_wet_mix", 0.30), 0.20, 0.40, 0.30),
        "denoise_reduction_db": clamp(read("denoise_reduction_db", 5), 3, 8, 5),
        "noise_profile_ms": clamp(read("noise_profile_ms", 350), 200, 500, 350),
        "noise_floor_dbfs": clamp(read("noise_floor_dbfs", -66), -72, -60, -66),
        "noise_floor_shape": shape,
        "timing_jitter_ms": clamp(read("timing_jitter_ms", 2.25), 1.5, 3, 2.25),
        "saturation_drive": clamp(read("saturation_drive", 0.10), 0, 0.25, 0.10),
        "multiband_compression": bool(read("multiband_compression", False)),
        "noise_floor_enabled": bool(read("noise_floor_enabled", True)),
        "timing_jitter_enabled": bool(read("timing_jitter_enabled", True)),
        "saturation_enabled": bool(read("saturation_enabled", True)),
    }


def role_from_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value).strip().lower())
    if any(x in name for x in ("vocal", "voice", "vox")):
        return "vocal"
    if any(x in name for x in ("drum", "percussion", "kick", "snare", "cymbal", "hihat", "hi-hat")):
        return "percussion"
    if any(x in name for x in ("guitar", "bass", "synth", "piano", "keys", "pad", "string", "instrumental", "music", "accompaniment", "other")):
        return "harmonic"
    return "other"


def align_audio(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if len(audio) == length:
        return audio
    out = np.zeros((length, 2), np.float32)
    out[: min(length, len(audio))] = audio[: min(length, len(audio))]
    return out


def _sample(audio: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.clip(positions, 0, max(0, len(audio) - 1.000001))
    lo = np.floor(positions).astype(np.int64)
    hi = np.minimum(lo + 1, len(audio) - 1)
    frac = (positions - lo).astype(np.float32)[:, None]
    return audio[lo] * (1 - frac) + audio[hi] * frac


def _vibrato(audio: np.ndarray, sr: int, rate: float, cents: float, phase: float = 0) -> np.ndarray:
    if cents <= 0 or len(audio) < 2:
        return audio.copy()
    excursion = (2 ** (cents / 1200) - 1) * sr / (2 * math.pi * rate)
    out = np.empty_like(audio)
    for start in range(0, len(audio), 262_144):
        stop = min(len(audio), start + 262_144)
        idx = np.arange(start, stop, dtype=np.float64)
        out[start:stop] = _sample(audio, idx + excursion * np.sin(2 * math.pi * rate * idx / sr + phase))
    return out


def _tremolo(audio: np.ndarray, sr: int, rate: float, depth: float, waveform: str) -> np.ndarray:
    if depth <= 0:
        return audio.copy()
    idx = np.arange(len(audio), dtype=np.float64)
    wave = np.sin(2 * math.pi * rate * idx / sr + math.pi / 3)
    if waveform == "soft_triangle":
        wave = (2 / math.pi) * np.arcsin(np.clip(wave, -1, 1))
    return (audio * (1 + depth / 100 * wave)[:, None]).astype(np.float32)


def _flanger(audio: np.ndarray, sr: int, delay_ms: float, rate: float, depth: float, feedback: float, wet: float) -> np.ndarray:
    if wet <= 0 or len(audio) < 2:
        return audio.copy()
    base = delay_ms * sr / 1000
    excursion = base * 0.35 * np.clip(depth / 10, 0, 1)
    tap1 = np.empty_like(audio)
    tap2 = np.empty_like(audio)
    for start in range(0, len(audio), 262_144):
        stop = min(len(audio), start + 262_144)
        idx = np.arange(start, stop, dtype=np.float64)
        delay = base + excursion * np.sin(2 * math.pi * rate * idx / sr)
        tap1[start:stop] = _sample(audio, idx - delay)
        tap2[start:stop] = _sample(audio, idx - 2 * delay)
    return (audio * (1 - wet) + (tap1 + feedback * tap2) * wet).astype(np.float32)


def _filter(audio: np.ndarray, sr: int, kind: str, hz: float) -> np.ndarray:
    if len(audio) < 64:
        return np.zeros_like(audio)
    sos = signal.butter(2, np.clip(hz, 10, sr * 0.45), btype=kind, fs=sr, output="sos")
    try:
        return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def _mid_emphasized_delta(reference: np.ndarray, effected: np.ndarray, sr: int) -> np.ndarray:
    delta = effected - reference
    low = _filter(delta, sr, "lowpass", 180)
    high = _filter(delta, sr, "highpass", 7000)
    return (reference + (delta - low - high) + 0.45 * (low + high)).astype(np.float32)


def core_cocktail(audio: np.ndarray, sr: int, intensity: float, role_multiplier: float, p: dict) -> tuple[np.ndarray, list[dict]]:
    scale = intensity * role_multiplier
    if scale <= 0:
        return audio.copy(), [{"operation": x, "bypassed": True} for x in ("vibrato", "tremolo", "vibrato", "flanger")]
    d1 = p["primary_vibrato_depth_cents"] * scale
    out = _mid_emphasized_delta(audio, _vibrato(audio, sr, p["primary_vibrato_rate_hz"], d1), sr)
    ops = [{"operation": "vibrato", "stage": "primary", "rate_hz": p["primary_vibrato_rate_hz"], "waveform": "sine", "effective_depth_cents": round(d1, 4), "role_multiplier": round(role_multiplier, 4)}]
    dt = p["tremolo_depth_percent"] * scale
    out = _mid_emphasized_delta(out, _tremolo(out, sr, p["tremolo_rate_hz"], dt, p["tremolo_waveform"]), sr)
    ops.append({"operation": "tremolo", "rate_hz": p["tremolo_rate_hz"], "waveform": p["tremolo_waveform"], "effective_depth_percent": round(dt, 4), "role_multiplier": round(role_multiplier, 4)})
    d2 = p["secondary_vibrato_depth_cents"] * scale
    out = _mid_emphasized_delta(out, _vibrato(out, sr, p["secondary_vibrato_rate_hz"], d2, math.pi / 2), sr)
    ops.append({"operation": "vibrato", "stage": "secondary", "rate_hz": p["secondary_vibrato_rate_hz"], "waveform": "sine", "effective_depth_cents": round(d2, 4), "role_multiplier": round(role_multiplier, 4)})
    fd = p["flanger_depth"] * scale
    wet = float(np.clip(p["flanger_wet_mix"] * scale, 0, 0.40))
    out = _mid_emphasized_delta(out, _flanger(out, sr, p["flanger_delay_ms"], p["flanger_modulation_rate_hz"], fd, p["flanger_feedback"], wet), sr)
    ops.append({"operation": "flanger", "delay_ms": p["flanger_delay_ms"], "modulation_rate_hz": p["flanger_modulation_rate_hz"], "effective_depth": round(fd, 4), "feedback": p["flanger_feedback"], "wet_mix": round(wet, 4), "role_multiplier": round(role_multiplier, 4)})
    return out, ops


def _frame_envelope(audio: np.ndarray, frame: int) -> np.ndarray:
    mono = np.mean(np.abs(audio), axis=1, dtype=np.float64)
    count = int(math.ceil(len(mono) / frame))
    padded = np.pad(mono, (0, count * frame - len(mono)))
    return np.max(padded.reshape(count, frame), axis=1)


def micro_timing_jitter(audio: np.ndarray, sr: int, maximum_ms: float, seed: int) -> tuple[np.ndarray, dict]:
    if maximum_ms <= 0 or len(audio) < sr // 4:
        return audio.copy(), {"operation": "micro_timing_jitter", "bypassed": True, "maximum_ms": round(maximum_ms, 4), "onset_count": 0}
    hop = max(64, int(sr * 0.005))
    env = _frame_envelope(audio, hop)
    onset = np.maximum(np.diff(env, prepend=env[0]), 0)
    peaks, props = signal.find_peaks(onset, distance=max(1, int(0.045 * sr / hop)), prominence=max(float(np.quantile(onset, 0.8)) * 0.5, 1e-8))
    if len(peaks) > 256:
        peaks = np.sort(peaks[np.argsort(props["prominences"])[-256:]])
    rng = np.random.default_rng(seed)
    control = np.zeros(len(env))
    ref = max(float(np.quantile(env, 0.95)), 1e-8)
    max_samples = maximum_ms * sr / 1000
    for index in peaks:
        control[index] = rng.uniform(-max_samples, max_samples) * np.clip(env[index] / ref, 0.15, 1)
    radius = max(1, int(0.012 * sr / hop))
    window = signal.windows.gaussian(radius * 8 + 1, std=max(1, radius * 1.5))
    smooth = signal.fftconvolve(control, window / max(window.max(), 1e-12), mode="same")
    positions = np.arange(len(audio), dtype=np.float64)
    offsets = np.interp(positions, np.arange(len(env)) * hop, smooth)
    return _sample(audio, positions - offsets), {"operation": "micro_timing_jitter", "bypassed": False, "maximum_ms": round(maximum_ms, 4), "onset_count": int(len(peaks)), "velocity_sensitive": True, "seed": seed}


def match_level(reference: np.ndarray, candidate: np.ndarray, maximum_db: float = 0.75) -> tuple[np.ndarray, dict]:
    requested = db(rms(reference) / max(rms(candidate), 1e-12))
    applied = float(np.clip(requested, -maximum_db, maximum_db))
    return (candidate * 10 ** (applied / 20)).astype(np.float32), {"operation": "level_preservation", "requested_adjustment_db": round(requested, 4), "applied_adjustment_db": round(applied, 4), "maximum_adjustment_db": maximum_db}


def saturation(audio: np.ndarray, drive: float) -> tuple[np.ndarray, dict]:
    if drive <= 0:
        return audio.copy(), {"operation": "low_drive_saturation", "bypassed": True, "drive": 0.0}
    gain = 1 + 2.5 * drive
    out = np.tanh(audio.astype(np.float64) * gain) / max(math.tanh(gain), 1e-12)
    out, level = match_level(audio, out.astype(np.float32), 0.5)
    return out, {"operation": "low_drive_saturation", "bypassed": False, "style": "tape_tube_soft_clip", "drive": round(drive, 4), "gain_compensated": True, "level": level}


def multiband_compress(audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    low = _filter(audio, sr, "lowpass", 180)
    high = _filter(audio, sr, "highpass", 5000)
    mid = audio - low - high
    bands = [
        linked_compressor(low, sr, threshold_db=-20, ratio=1.25, attack_ms=35, release_ms=220, knee_db=6),
        linked_compressor(mid, sr, threshold_db=-18, ratio=1.20, attack_ms=25, release_ms=180, knee_db=6),
        linked_compressor(high, sr, threshold_db=-22, ratio=1.15, attack_ms=12, release_ms=150, knee_db=5),
    ]
    out, level = match_level(audio, sum(bands), 0.5)
    return out, {"operation": "gentle_multiband_compression", "bypassed": False, "crossovers_hz": [180, 5000], "ratios": [1.25, 1.20, 1.15], "gain_compensated": True, "level": level}


def restore_noise_floor(audio: np.ndarray, sr: int, level_dbfs: float, shape: str, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(audio.shape).astype(np.float64)
    if shape == "tape_hiss":
        shaped = _filter(white.astype(np.float32), sr, "highpass", 2500)
        shaped = _filter(shaped, sr, "lowpass", 15000).astype(np.float64)
    else:
        shaped = signal.lfilter([0.049922035, 0.095993537, 0.050612699, -0.004408786], [1.0, -2.494956002, 2.017265875, -0.522189400], white, axis=0)
        shape = "pink"
    shaped *= 10 ** (level_dbfs / 20) / max(float(np.sqrt(np.mean(shaped**2) + 1e-20)), 1e-12)
    return (audio + shaped).astype(np.float32), {"operation": "noise_floor_restoration", "bypassed": False, "level_dbfs": round(level_dbfs, 3), "shape": shape, "seed": seed, "position": "after_cocktail_before_denoise"}


def _quiet_profile(audio: np.ndarray, sr: int, duration_ms: float) -> tuple[np.ndarray, dict]:
    target = max(1, int(sr * duration_ms / 1000))
    if len(audio) <= target:
        return audio, {"start_seconds": 0.0, "duration_ms": round(len(audio) / sr * 1000, 3)}
    frame = max(1, int(sr * 0.05))
    mono = np.mean(audio, axis=1, dtype=np.float64)
    framed = mono[: len(mono) // frame * frame].reshape(-1, frame)
    levels = np.sqrt(np.mean(framed**2, axis=1) + 1e-20)
    width = max(1, int(math.ceil(target / frame)))
    start = int(np.argmin(np.convolve(levels, np.ones(width) / width, mode="valid"))) * frame
    return audio[start : start + target], {"start_seconds": round(start / sr, 6), "duration_ms": round(target / sr * 1000, 3), "selection": "quietest_post_modulation_window"}


def adaptive_music_denoise(audio: np.ndarray, sr: int, maximum_reduction_db: float, profile_ms: float) -> tuple[np.ndarray, dict]:
    profile, profile_report = _quiet_profile(audio, sr, profile_ms)
    nperseg, noverlap = 2048, 1536
    floor_gain = 10 ** (-maximum_reduction_db / 20)
    out = np.empty_like(audio)
    means = []
    for channel in range(audio.shape[1]):
        _, _, noise_spec = signal.stft(profile[:, channel], fs=sr, nperseg=min(nperseg, max(64, len(profile))), noverlap=min(noverlap, max(0, len(profile) // 2)), boundary="zeros", padded=True)
        noise = np.median(np.abs(noise_spec), axis=1)
        _, _, spec = signal.stft(audio[:, channel], fs=sr, nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True)
        mag = np.abs(spec)
        if len(noise) != len(mag):
            noise = np.interp(np.linspace(0, 1, len(mag)), np.linspace(0, 1, len(noise)), noise)
        mask = np.clip(1 - 0.58 * (noise[:, None] / np.maximum(mag, 1e-10)) ** 2, floor_gain, 1)
        flux = np.zeros(mask.shape[1])
        if mask.shape[1] > 1:
            flux[1:] = np.mean(np.maximum(np.diff(mag, axis=1), 0), axis=0)
        transient = np.clip(flux / max(float(np.quantile(flux, 0.9)), 1e-12), 0, 1)
        mask = signal.medfilt2d(mask * (1 - 0.72 * transient[None, :]) + 0.72 * transient[None, :], kernel_size=(3, 3))
        attack = math.exp(-(nperseg - noverlap) / max(1, sr * 0.010))
        release = math.exp(-(nperseg - noverlap) / max(1, sr * 0.190))
        state = np.ones(mask.shape[0])
        smooth = np.empty_like(mask)
        for frame_index in range(mask.shape[1]):
            target = mask[:, frame_index]
            coefficient = np.where(target < state, attack, release)
            state = coefficient * state + (1 - coefficient) * target
            smooth[:, frame_index] = state
        _, recon = signal.istft(spec * smooth, fs=sr, nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=True)
        out[:, channel] = 0.90 * recon[: len(audio)] + 0.10 * audio[:, channel]
        means.append(float(np.average(np.maximum(-20 * np.log10(np.maximum(smooth, 1e-12)), 0), weights=np.maximum(mag, 1e-12))))
    return out, {"operation": "constrained_post_cocktail_denoise", "type": "adaptive_music_spectral", "order_requirement": "post_cocktail_only", "maximum_reduction_db": round(maximum_reduction_db, 4), "exact_mean_reduction_db": round(float(np.mean(means)), 4), "noise_profile": profile_report, "attack_ms": 10.0, "release_ms": 190.0, "transient_protection": True, "dry_microvariation_preservation_mix": 0.10, "spectral_uniformity_restoration": False}


def apply_headroom(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    maximum = peak(audio)
    gain = 1.0 if maximum <= 0.985 else 0.985 / max(maximum, 1e-12)
    return (audio * gain).astype(np.float32), {"operation": "non_limiting_peak_safety", "gain_db": round(db(gain), 4), "limiter_used": False}


def reconstruction_report(source: np.ndarray, groups: list[dict]) -> dict:
    if not groups:
        return {"accepted": False, "reason": "no_stems", "waveform_correlation": 0.0, "level_matched_residual_relative_db": 0.0}
    candidate = np.sum([x["audio"] for x in groups], axis=0, dtype=np.float64)
    a = np.mean(source, axis=1, dtype=np.float64); b = np.mean(candidate, axis=1, dtype=np.float64)
    a -= a.mean(); b -= b.mean()
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20))
    matched = candidate * rms(source) / max(rms(candidate), 1e-12)
    residual = db(rms(matched - source) / max(rms(source), 1e-12))
    return {"accepted": corr >= 0.985 and residual <= -24, "waveform_correlation": round(corr, 7), "level_matched_residual_relative_db": round(residual, 4), "thresholds": {"minimum_waveform_correlation": 0.985, "maximum_residual_relative_db": -24.0}}


def _spectral_cosine(reference: np.ndarray, candidate: np.ndarray, sr: int) -> float:
    length = min(len(reference), len(candidate), sr * 90)
    if length < 512:
        return 1.0
    ref = np.mean(reference[:length], axis=1); can = np.mean(candidate[:length], axis=1)
    _, _, rs = signal.stft(ref, fs=sr, nperseg=2048, noverlap=1536)
    _, _, cs = signal.stft(can, fs=sr, nperseg=2048, noverlap=1536)
    rm = np.mean(np.abs(rs), axis=1); cm = np.mean(np.abs(cs), axis=1)
    return float(np.dot(rm, cm) / (np.linalg.norm(rm) * np.linalg.norm(cm) + 1e-20))


def _envelope_metrics(reference: np.ndarray, candidate: np.ndarray, sr: int) -> dict:
    frame = max(64, int(sr * 0.01))
    ref = _frame_envelope(reference, frame); can = _frame_envelope(candidate, frame)
    length = min(len(ref), len(can))
    if length < 2:
        return {"correlation": 1.0, "transient_retention": 1.0, "pumping_index": 0.0}
    ref = ref[:length]; can = can[:length]
    rc = ref - ref.mean(); cc = can - can.mean()
    corr = float(np.dot(rc, cc) / (np.linalg.norm(rc) * np.linalg.norm(cc) + 1e-20))
    ro = np.maximum(np.diff(ref, prepend=ref[0]), 0); co = np.maximum(np.diff(can, prepend=can[0]), 0)
    active = ro >= max(float(np.quantile(ro, 0.9)), 1e-10)
    retention = float(np.median(co[active] / np.maximum(ro[active], 1e-10))) if np.any(active) else 1.0
    pumping = float(np.std(signal.detrend(can / np.maximum(ref, 1e-8))))
    return {"correlation": round(corr, 6), "transient_retention": round(float(np.clip(retention, 0, 2)), 6), "pumping_index": round(pumping, 6)}


def artifact_assessment(reference: np.ndarray, candidate: np.ndarray, sr: int, intensity: float, stage: str = "dsp") -> dict:
    comparison = compare_audio_quality(reference, candidate, sr)
    envelope = _envelope_metrics(reference, candidate, sr)
    spectral = _spectral_cosine(reference, candidate, sr)
    before = analyze_array(reference, sr); after = analyze_array(candidate, sr)
    crest_delta = float(after["crest_factor_db"]) - float(before["crest_factor_db"])
    reasons = []
    if comparison["non_finite_sample_count"]: reasons.append("non_finite_samples")
    if int(after.get("clipped_sample_count", 0)) > 0: reasons.append("clipped_samples")
    floor = (0.90 if stage == "dsp" else 0.82) - max(0, intensity - 1) * 0.04
    if float(comparison["waveform_correlation"]) < floor: reasons.append("clarity_loss_or_excess_source_deviation")
    if float(comparison["derivative"]["rms_delta_db"]) > 5: reasons.append("metallic_or_derivative_artifact")
    if float(comparison["derivative"]["p999_delta_db"]) > 6: reasons.append("transient_or_click_growth")
    if float(comparison["high_band_energy_ratio"]["delta_db"]) > 4: reasons.append("aliasing_or_high_frequency_growth")
    if spectral < (0.94 if stage == "dsp" else 0.88): reasons.append("spectral_clarity_degradation")
    if float(envelope["correlation"]) < 0.88: reasons.append("pumping_or_envelope_damage")
    if float(envelope["transient_retention"]) < 0.70: reasons.append("transient_loss")
    if float(envelope["pumping_index"]) > 0.20: reasons.append("pumping")
    if crest_delta < -3: reasons.append("dynamic_clarity_loss")
    warble = float(comparison["waveform_correlation"]) < 0.92 and float(envelope["correlation"]) >= 0.90 and intensity > 1.0
    if warble: reasons.append("audible_warble_risk")
    return {"accepted": not reasons, "stage": stage, "reasons": reasons, "comparison": comparison, "envelope": envelope, "spectral_cosine_similarity": round(spectral, 7), "crest_factor_delta_db": round(crest_delta, 4), "detectors": {"warble_risk": warble, "pumping_risk": any("pumping" in x for x in reasons), "metallic_risk": any("metallic" in x for x in reasons), "clarity_loss_risk": any("clarity" in x for x in reasons), "aliasing_risk": any("aliasing" in x for x in reasons), "transient_loss_risk": "transient_loss" in reasons}}


def render_dsp_role(audio: np.ndarray, sr: int, role: str, intensity: float, passes: int, second_scale: float, parameters: dict, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    role_multiplier = ROLE_MULTIPLIERS.get(role, ROLE_MULTIPLIERS["other"])
    processed = audio.astype(np.float32)
    pre_denoise = processed.copy()
    reports = []
    for pass_index in range(passes):
        pass_scale = 1.0 if pass_index == 0 else second_scale
        pass_intensity = intensity * pass_scale
        processed, operations = core_cocktail(processed, sr, pass_intensity, role_multiplier, parameters)
        timing_scale = {"vocal": 1.0, "harmonic": 0.75, "percussion": 0.45, "other": 0.65}.get(role, 0.65)
        if parameters["timing_jitter_enabled"] and pass_intensity > 0:
            processed, timing = micro_timing_jitter(processed, sr, parameters["timing_jitter_ms"] * pass_intensity * timing_scale, seed + pass_index * 101)
        else:
            timing = {"operation": "micro_timing_jitter", "bypassed": True, "maximum_ms": 0.0, "onset_count": 0}
        operations.append(timing)
        if parameters["saturation_enabled"]:
            processed, sat = saturation(processed, parameters["saturation_drive"] * pass_intensity)
        else:
            sat = {"operation": "low_drive_saturation", "bypassed": True, "drive": 0.0}
        operations.append(sat)
        if parameters["multiband_compression"] and pass_intensity > 0:
            processed, multiband = multiband_compress(processed, sr)
        else:
            multiband = {"operation": "gentle_multiband_compression", "bypassed": True}
        operations.append(multiband)
        if parameters["noise_floor_enabled"] and pass_intensity > 0:
            processed, floor = restore_noise_floor(processed, sr, parameters["noise_floor_dbfs"], parameters["noise_floor_shape"], seed + pass_index * 313)
        else:
            floor = {"operation": "noise_floor_restoration", "bypassed": True, "level_dbfs": parameters["noise_floor_dbfs"]}
        operations.append(floor)
        pre_denoise = processed.copy()
        processed, denoise = adaptive_music_denoise(processed, sr, parameters["denoise_reduction_db"], parameters["noise_profile_ms"])
        operations.append(denoise)
        reports.append({"pass": pass_index + 1, "intensity": round(pass_intensity, 4), "scale": round(pass_scale, 4), "operations": operations})
    return processed, pre_denoise, {"role": role, "role_multiplier": role_multiplier, "passes": reports, "operations": [op for item in reports for op in item["operations"]]}


def processing_order() -> list[str]:
    return [
        "primary_vibrato_14_16hz_default15_depth8_12c_default10",
        "tremolo_12_16hz_default15_depth6_12pct_default10",
        "secondary_vibrato_8_11hz_default10_depth6_10c_default9",
        "flanger_delay0.8_1.2ms_default1_rate0.6_1.1hz_default0.9",
        "optional_timing_saturation_multiband_and_noise_floor",
        "constrained_adaptive_denoise_after_modulation",
        "optional_neural_vocoder_after_denoise",
    ]
