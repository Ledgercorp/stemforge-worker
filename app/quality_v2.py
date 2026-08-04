from __future__ import annotations

import math

import numpy as np
from scipy import signal


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-20))


def _db(value: float, floor: float = -180.0) -> float:
    return floor if value <= 0.0 else 20.0 * math.log10(value)


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)
    return np.mean(audio, axis=1, dtype=np.float64)


def _finite_count(audio: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isfinite(audio)))


def _waveform_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = _mono(reference)
    can = _mono(candidate)
    length = min(len(ref), len(can))
    if length < 2:
        return 0.0
    ref = ref[:length] - np.mean(ref[:length])
    can = can[:length] - np.mean(can[:length])
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(can)) + 1e-20
    return float(np.dot(ref, can) / denominator)


def _level_match(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, float]:
    ref_rms = _rms(reference)
    can_rms = _rms(candidate)
    gain = ref_rms / max(can_rms, 1e-12)
    return candidate.astype(np.float64) * gain, float(gain)


def _derivative_metrics(audio: np.ndarray) -> dict:
    if len(audio) < 2:
        return {
            "rms": 0.0,
            "p999": 0.0,
            "maximum": 0.0,
            "robust_outlier_count": 0,
            "robust_threshold": 0.0,
        }
    derivative = np.max(np.abs(np.diff(audio.astype(np.float64), axis=0)), axis=1)
    median = float(np.median(derivative))
    mad = float(np.median(np.abs(derivative - median)))
    p999 = float(np.quantile(derivative, 0.999))
    # A floor prevents normal transients from being called clicks while the
    # robust term catches dense low-level discontinuities and zipper noise.
    threshold = max(0.08, median + 35.0 * max(mad, 1e-9), p999 * 1.75)
    return {
        "rms": float(np.sqrt(np.mean(np.square(derivative)) + 1e-20)),
        "p999": p999,
        "maximum": float(np.max(derivative)),
        "robust_outlier_count": int(np.count_nonzero(derivative > threshold)),
        "robust_threshold": threshold,
    }


def _band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    mono = _mono(audio)
    if len(mono) < 256:
        return 0.0
    if len(mono) > sample_rate * 300:
        step = max(1, len(mono) // (sample_rate * 300))
        mono = mono[::step]
        sample_rate = max(1, sample_rate // step)
    nperseg = min(8192, max(512, len(mono)))
    frequencies, psd = signal.welch(mono, sample_rate, nperseg=nperseg)
    upper = min(float(high_hz), sample_rate * 0.499)
    mask = (frequencies >= float(low_hz)) & (frequencies < upper)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], frequencies[mask]) + 1e-20)


def compare_audio_quality(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict:
    length = min(len(reference), len(candidate))
    reference = reference[:length].astype(np.float64)
    candidate = candidate[:length].astype(np.float64)

    matched, match_gain = _level_match(reference, candidate)
    residual = matched - reference
    residual_relative_db = _db(_rms(residual) / max(_rms(reference), 1e-12))

    ref_derivative = _derivative_metrics(reference)
    can_derivative = _derivative_metrics(candidate)
    derivative_rms_delta_db = _db(
        can_derivative["rms"] / max(ref_derivative["rms"], 1e-12)
    )
    derivative_p999_delta_db = _db(
        can_derivative["p999"] / max(ref_derivative["p999"], 1e-12)
    )

    ref_high = _band_energy(reference, sample_rate, 8000.0, 20000.0)
    can_high = _band_energy(candidate, sample_rate, 8000.0, 20000.0)
    ref_total = _band_energy(reference, sample_rate, 20.0, 20000.0)
    can_total = _band_energy(candidate, sample_rate, 20.0, 20000.0)
    ref_high_ratio = ref_high / max(ref_total, 1e-20)
    can_high_ratio = can_high / max(can_total, 1e-20)
    high_band_ratio_delta_db = _db(can_high_ratio / max(ref_high_ratio, 1e-20))

    return {
        "waveform_correlation": round(_waveform_correlation(reference, candidate), 7),
        "level_match_gain_db": round(_db(match_gain), 4),
        "level_matched_residual_relative_db": round(residual_relative_db, 4),
        "non_finite_sample_count": _finite_count(candidate),
        "derivative": {
            "reference_rms": round(ref_derivative["rms"], 9),
            "candidate_rms": round(can_derivative["rms"], 9),
            "rms_delta_db": round(derivative_rms_delta_db, 4),
            "reference_p999": round(ref_derivative["p999"], 9),
            "candidate_p999": round(can_derivative["p999"], 9),
            "p999_delta_db": round(derivative_p999_delta_db, 4),
            "reference_maximum": round(ref_derivative["maximum"], 9),
            "candidate_maximum": round(can_derivative["maximum"], 9),
            "reference_outlier_count": ref_derivative["robust_outlier_count"],
            "candidate_outlier_count": can_derivative["robust_outlier_count"],
        },
        "high_band_energy_ratio": {
            "reference": round(ref_high_ratio, 9),
            "candidate": round(can_high_ratio, 9),
            "delta_db": round(high_band_ratio_delta_db, 4),
        },
    }


def quality_gate(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    candidate_metrics: dict,
    *,
    mode: str,
) -> dict:
    comparison = compare_audio_quality(reference, candidate, sample_rate)
    reasons: list[str] = []

    if comparison["non_finite_sample_count"]:
        reasons.append("non_finite_samples")
    if int(candidate_metrics.get("clipped_sample_count", 0)) > 0:
        reasons.append("clipped_samples")
    if float(candidate_metrics.get("true_peak_dbtp", 0.0)) > -0.75:
        reasons.append("true_peak_margin")

    correlation_floor = 0.965 if mode == "master_anchored_delta" else 0.72
    if float(comparison["waveform_correlation"]) < correlation_floor:
        reasons.append("low_reference_correlation")

    derivative = comparison["derivative"]
    if float(derivative["rms_delta_db"]) > 4.5:
        reasons.append("excess_derivative_energy")
    if float(derivative["p999_delta_db"]) > 5.5:
        reasons.append("transient_or_click_growth")
    allowed_outliers = int(derivative["reference_outlier_count"]) + 8
    if int(derivative["candidate_outlier_count"]) > allowed_outliers:
        reasons.append("new_waveform_discontinuities")

    if float(comparison["high_band_energy_ratio"]["delta_db"]) > 4.0:
        reasons.append("high_frequency_noise_growth")

    return {
        "accepted": not reasons,
        "reasons": reasons,
        "mode": mode,
        "comparison": comparison,
    }
