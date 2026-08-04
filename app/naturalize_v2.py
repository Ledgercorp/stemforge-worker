from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal

from app.analysis_v2 import analyze_array
from app.audio_core import db, ensure_pcm_wav, load_audio, peak, rms, write_audio
from app.quality_v2 import compare_audio_quality
from app.storage import materialize_input, publish_files

NATURALIZE_VERSION = "1.0"
NATURALIZE_LABEL = "Naturalness / Authenticity enhancement"
NOMINAL_COCKTAIL = {
    "vibrato_1": {"rate_hz": 15.0, "depth": 10.0},
    "tremolo": {"rate_hz": 15.0, "depth": 10.0},
    "vibrato_2": {"rate_hz": 10.0, "depth": 10.0},
    "flanger": {"delay_ms": 1.0, "modulation": 0.90, "depth": 9.0},
}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "untitled"


def _resolve_intensity(payload: dict) -> float:
    value = float(payload.get("intensity", payload.get("depth", 0.35)))
    if 1.0 < value <= 100.0:
        value /= 100.0
    if not np.isfinite(value):
        raise ValueError("Naturalize intensity must be finite.")
    return float(np.clip(value, 0.0, 1.0))


def _align(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio.astype(np.float32, copy=False)
    output = np.zeros((length, audio.shape[1]), dtype=np.float32)
    copy_length = min(length, len(audio))
    output[:copy_length] = audio[:copy_length]
    return output


def _sample(audio: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.clip(positions, 0.0, max(0.0, len(audio) - 1.000001))
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(audio) - 1)
    fraction = (positions - lower).astype(np.float32)[:, None]
    return audio[lower] * (1.0 - fraction) + audio[upper] * fraction


def _vibrato(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    depth_cents: float,
    phase: float = 0.0,
) -> np.ndarray:
    if depth_cents <= 0.0:
        return audio.copy()
    pitch_delta = 2.0 ** (depth_cents / 1200.0) - 1.0
    excursion = pitch_delta * sample_rate / (2.0 * math.pi * rate_hz)
    output = np.empty_like(audio, dtype=np.float32)
    for start in range(0, len(audio), 262_144):
        stop = min(len(audio), start + 262_144)
        indices = np.arange(start, stop, dtype=np.float64)
        positions = indices + excursion * np.sin(
            2.0 * math.pi * rate_hz * indices / sample_rate + phase
        )
        output[start:stop] = _sample(audio, positions)
    return output


def _tremolo(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    depth_percent: float,
) -> np.ndarray:
    if depth_percent <= 0.0:
        return audio.copy()
    indices = np.arange(len(audio), dtype=np.float64)
    gain = 1.0 + (depth_percent / 100.0) * np.sin(
        2.0 * math.pi * rate_hz * indices / sample_rate + math.pi / 3.0
    )
    return (audio * gain[:, None]).astype(np.float32)


def _flanger(
    audio: np.ndarray,
    sample_rate: int,
    delay_ms: float,
    modulation: float,
    wet_percent: float,
) -> np.ndarray:
    if wet_percent <= 0.0:
        return audio.copy()
    wet_mix = float(np.clip(wet_percent / 100.0, 0.0, 0.25))
    base_delay = delay_ms * sample_rate / 1000.0
    modulation_samples = base_delay * float(np.clip(modulation, 0.0, 1.0))
    delayed = np.empty_like(audio, dtype=np.float32)
    for start in range(0, len(audio), 262_144):
        stop = min(len(audio), start + 262_144)
        indices = np.arange(start, stop, dtype=np.float64)
        delay = base_delay + modulation_samples * np.sin(
            2.0 * math.pi * 0.23 * indices / sample_rate
        )
        delayed[start:stop] = _sample(audio, indices - delay)
    return (audio * (1.0 - wet_mix) + delayed * wet_mix).astype(np.float32)


def _light_denoise(
    audio: np.ndarray,
    sample_rate: int,
    intensity: float,
) -> tuple[np.ndarray, dict]:
    if intensity <= 0.0:
        return audio.copy(), {
            "operation": "light_post_modulation_denoise",
            "order_requirement": "after_all_modulation",
            "maximum_reduction_db": 0.0,
            "noise_profile_percentile": 12.0,
            "bypassed": True,
        }
    maximum_reduction_db = 0.75 + 1.25 * intensity
    floor_gain = 10.0 ** (-maximum_reduction_db / 20.0)
    strength = 0.18 + 0.32 * intensity
    output = np.empty_like(audio, dtype=np.float32)
    for channel in range(audio.shape[1]):
        _, _, spectrum = signal.stft(
            audio[:, channel],
            fs=sample_rate,
            nperseg=2048,
            noverlap=1536,
            boundary="zeros",
            padded=True,
        )
        magnitude = np.abs(spectrum)
        noise = np.quantile(magnitude, 0.12, axis=1, keepdims=True)
        mask = np.clip(
            1.0 - strength * noise / np.maximum(magnitude, 1e-10),
            floor_gain,
            1.0,
        )
        mask = signal.medfilt2d(mask, kernel_size=(3, 3))
        _, reconstructed = signal.istft(
            spectrum * mask,
            fs=sample_rate,
            nperseg=2048,
            noverlap=1536,
            input_onesided=True,
            boundary=True,
        )
        output[:, channel] = reconstructed[: len(audio)]
    return output, {
        "operation": "light_post_modulation_denoise",
        "order_requirement": "after_all_modulation",
        "maximum_reduction_db": round(maximum_reduction_db, 3),
        "noise_profile_percentile": 12.0,
        "bypassed": False,
    }


def _match_level(
    reference: np.ndarray,
    candidate: np.ndarray,
    maximum_db: float,
) -> tuple[np.ndarray, dict]:
    requested_db = db(rms(reference) / max(rms(candidate), 1e-12))
    applied_db = float(np.clip(requested_db, -maximum_db, maximum_db))
    output = candidate * 10.0 ** (applied_db / 20.0)
    return output.astype(np.float32), {
        "operation": "level_preservation",
        "requested_adjustment_db": round(requested_db, 4),
        "applied_adjustment_db": round(applied_db, 4),
        "maximum_adjustment_db": maximum_db,
    }


def _headroom(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    maximum = peak(audio)
    gain = 1.0 if maximum <= 0.985 else 0.985 / max(maximum, 1e-12)
    return (audio * gain).astype(np.float32), {
        "operation": "non_limiting_peak_safety",
        "gain_db": round(db(gain), 4),
        "limiter_used": False,
    }


def _cocktail(
    audio: np.ndarray,
    sample_rate: int,
    intensity: float,
    include_denoise: bool,
) -> tuple[np.ndarray, list[dict]]:
    processed = _vibrato(audio, sample_rate, 15.0, 10.0 * intensity)
    operations = [
        {
            "operation": "vibrato",
            "rate_hz": 15.0,
            "nominal_depth": 10.0,
            "effective_depth_cents": round(10.0 * intensity, 4),
        }
    ]
    processed = _tremolo(processed, sample_rate, 15.0, 10.0 * intensity)
    operations.append(
        {
            "operation": "tremolo",
            "rate_hz": 15.0,
            "nominal_depth": 10.0,
            "effective_depth_percent": round(10.0 * intensity, 4),
        }
    )
    processed = _vibrato(
        processed,
        sample_rate,
        10.0,
        10.0 * intensity,
        math.pi / 2.0,
    )
    operations.append(
        {
            "operation": "vibrato",
            "rate_hz": 10.0,
            "nominal_depth": 10.0,
            "effective_depth_cents": round(10.0 * intensity, 4),
        }
    )
    processed = _flanger(processed, sample_rate, 1.0, 0.90, 9.0 * intensity)
    operations.append(
        {
            "operation": "flanger",
            "delay_ms": 1.0,
            "modulation": 0.90,
            "nominal_depth": 9.0,
            "effective_wet_percent": round(9.0 * intensity, 4),
        }
    )
    if include_denoise:
        processed, denoise = _light_denoise(processed, sample_rate, intensity)
        operations.append(denoise)
    return processed, operations


def _reconstruction(
    source: np.ndarray,
    vocal: np.ndarray,
    instrumental: np.ndarray,
) -> dict:
    candidate = vocal + instrumental
    source_mono = np.mean(source, axis=1, dtype=np.float64)
    candidate_mono = np.mean(candidate, axis=1, dtype=np.float64)
    source_mono -= np.mean(source_mono)
    candidate_mono -= np.mean(candidate_mono)
    denominator = np.linalg.norm(source_mono) * np.linalg.norm(candidate_mono) + 1e-20
    correlation = float(np.dot(source_mono, candidate_mono) / denominator)
    matched = candidate * (rms(source) / max(rms(candidate), 1e-12))
    residual_db = db(rms(matched - source) / max(rms(source), 1e-12))
    return {
        "accepted": correlation >= 0.985 and residual_db <= -24.0,
        "waveform_correlation": round(correlation, 7),
        "level_matched_residual_relative_db": round(residual_db, 4),
        "thresholds": {
            "minimum_waveform_correlation": 0.985,
            "maximum_residual_relative_db": -24.0,
        },
    }


def naturalize_arrays(
    source: np.ndarray,
    sample_rate: int,
    *,
    mode: str = "quick",
    intensity: float = 0.35,
    vocal: np.ndarray | None = None,
    instrumental: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    mode = str(mode).lower()
    if mode not in {"quick", "surgical"}:
        raise ValueError("Naturalize mode must be quick or surgical.")
    intensity = float(np.clip(intensity, 0.0, 1.0))
    reconstruction = None
    anchor = None

    if mode == "quick":
        processed, operations = _cocktail(source, sample_rate, intensity, True)
    else:
        if vocal is None:
            raise ValueError("Surgical Naturalize requires a vocal stem.")
        vocal = _align(vocal, len(source))
        if instrumental is None:
            instrumental = source - vocal
            anchor = {
                "instrumental_source": "master_minus_vocal",
                "reason": "No instrumental stem was supplied.",
            }
        else:
            instrumental = _align(instrumental, len(source))
            reconstruction = _reconstruction(source, vocal, instrumental)
            if not reconstruction["accepted"]:
                instrumental = source - vocal
                anchor = {
                    "instrumental_source": "master_minus_vocal",
                    "reason": "Supplied stems failed the reconstruction gate.",
                }
            else:
                anchor = {
                    "instrumental_source": "supplied_instrumental_stem",
                    "reason": "Supplied stems passed the reconstruction gate.",
                }

        vocal_intensity = min(1.0, intensity * 1.25)
        instrumental_intensity = intensity * 0.35
        vocal_processed, vocal_chain = _cocktail(
            vocal,
            sample_rate,
            vocal_intensity,
            False,
        )
        instrumental_processed, instrumental_chain = _cocktail(
            instrumental,
            sample_rate,
            instrumental_intensity,
            False,
        )
        instrumental_processed, instrumental_level = _match_level(
            instrumental,
            instrumental_processed,
            0.5,
        )
        processed = vocal_processed + instrumental_processed
        processed, final_denoise = _light_denoise(
            processed,
            sample_rate,
            intensity,
        )
        operations = [
            {
                "operation": "surgical_vocal_naturalize",
                "intensity": round(vocal_intensity, 4),
                "chain": vocal_chain,
            },
            {
                "operation": "surgical_instrumental_naturalize",
                "intensity": round(instrumental_intensity, 4),
                "normalization": instrumental_level,
                "chain": instrumental_chain,
            },
            final_denoise,
        ]

    processed, level_report = _match_level(source, processed, 0.75)
    processed, peak_report = _headroom(processed)
    operations.extend([level_report, peak_report])
    before = analyze_array(source, sample_rate)
    after = analyze_array(processed, sample_rate)
    comparison = compare_audio_quality(source, processed, sample_rate)
    correlation_floor = 0.985 - 0.08 * intensity
    reasons: list[str] = []
    if comparison["non_finite_sample_count"]:
        reasons.append("non_finite_samples")
    if after["clipped_sample_count"] > 0:
        reasons.append("clipped_samples")
    if float(comparison["waveform_correlation"]) < correlation_floor:
        reasons.append("excess_source_deviation")
    if float(comparison["derivative"]["rms_delta_db"]) > 3.5 + 2.0 * intensity:
        reasons.append("excess_derivative_growth")
    high_delta = float(comparison["high_band_energy_ratio"]["delta_db"])
    if np.isfinite(high_delta) and high_delta > 3.0:
        reasons.append("high_frequency_artifact_growth")

    report = {
        "status": "completed" if not reasons else "rejected",
        "engine": f"StemForge Naturalize pipeline v{NATURALIZE_VERSION}",
        "label": NATURALIZE_LABEL,
        "mode": mode,
        "intensity": round(intensity, 4),
        "non_destructive": True,
        "processing_order": [
            "vibrato_15hz_depth_10",
            "tremolo_15hz_depth_10",
            "vibrato_10hz_depth_10",
            "flanger_1ms_modulation_0.90_depth_9",
            "light_denoise_after_modulation",
        ],
        "nominal_cocktail": NOMINAL_COCKTAIL,
        "technical_rationale": {
            "vibrato": "Restores subtle pitch micro-movement.",
            "tremolo": "Restores organic amplitude variation over time.",
            "flanger": "Adds low-level phase movement associated with real recording paths.",
            "denoise_order": "Cleanup occurs after modulation so introduced micro-variations are retained.",
        },
        "operations": operations,
        "reconstruction_gate": reconstruction,
        "anchor_decision": anchor,
        "before": before,
        "after": after,
        "naturalness_characteristics": {
            "waveform_correlation": comparison["waveform_correlation"],
            "level_matched_residual_relative_db": comparison[
                "level_matched_residual_relative_db"
            ],
            "derivative_rms_delta_db": comparison["derivative"]["rms_delta_db"],
            "high_band_energy_ratio_delta_db": comparison[
                "high_band_energy_ratio"
            ]["delta_db"],
            "crest_factor_delta_db": round(
                float(after["crest_factor_db"]) - float(before["crest_factor_db"]),
                4,
            ),
        },
        "safety": {
            "accepted": not reasons,
            "reasons": reasons,
            "minimum_waveform_correlation": round(correlation_floor, 4),
            "comparison": comparison,
        },
        "honesty_note": (
            "Naturalize is a fidelity and authenticity enhancement. It does not "
            "identify, conceal, or guarantee the origin of audio and is not a detection tool."
        ),
    }
    return processed.astype(np.float32), report


def _stem_specs(payload: dict) -> list[dict]:
    raw = payload.get("stems") or []
    if isinstance(raw, dict):
        raw = [{"name": name, "url": value} for name, value in raw.items()]
    output = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            output.append({"name": f"stem_{index + 1}", "url": item})
        elif isinstance(item, dict):
            output.append(
                {
                    "name": str(
                        item.get("name")
                        or item.get("role")
                        or f"stem_{index + 1}"
                    ),
                    **item,
                }
            )
    return output


def _materialize_stem(spec: dict, destination: Path) -> Path:
    proxy = {}
    for suffix in ("url", "storage_key", "base64", "path", "size_bytes", "sha256"):
        if spec.get(suffix) is not None:
            proxy[f"stem_{suffix}"] = spec[suffix]
    return materialize_input(proxy, "stem", destination)


def _combine(
    paths: list[Path],
    length: int,
    sample_rate: int,
    root: Path,
) -> np.ndarray | None:
    combined = None
    for index, path in enumerate(paths):
        pcm = ensure_pcm_wav(
            path,
            root / f"stem_{index:02d}.wav",
            sample_rate=sample_rate,
        )
        audio, _ = load_audio(pcm)
        audio = _align(audio, length)
        combined = audio if combined is None else combined + audio
    return combined


def _separate(
    source: Path,
    job_dir: Path,
    model: str,
    timeout_seconds: int,
) -> tuple[Path, Path, dict]:
    output_root = job_dir / "demucs"
    command = [
        "python",
        "-m",
        "demucs.separate",
        "-n",
        model,
        "--two-stems",
        "vocals",
        "-o",
        str(output_root),
        str(source),
    ]
    try:
        proc = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Demucs is not installed in the worker image.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Demucs separation failed: {exc.stderr[-2000:]}") from exc
    vocals = next(iter(output_root.rglob("vocals.wav")), None)
    instrumental = next(iter(output_root.rglob("no_vocals.wav")), None)
    if vocals is None or instrumental is None:
        raise RuntimeError("Demucs completed without both vocal and instrumental stems.")
    return vocals, instrumental, {
        "engine": "Demucs two-stem separation",
        "model": model,
        "estimated_stems": True,
        "stdout_tail": proc.stdout[-1000:],
    }


def naturalize_audio_job(payload: dict) -> dict:
    artist = str(payload.get("artist") or "sounddecay")
    song = str(payload.get("song") or "untitled")
    requested_mode = str(payload.get("mode") or "auto").lower()
    if requested_mode not in {"auto", "quick", "surgical"}:
        return {
            "status": "rejected",
            "reason": "mode must be auto, quick, or surgical.",
        }
    intensity = _resolve_intensity(payload)
    job_dir = Path(payload.get("job_dir") or "/tmp/stemforge_naturalize_v2")
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    source_input = materialize_input(payload, "audio", job_dir / "source_input")
    source_pcm = ensure_pcm_wav(source_input, job_dir / "source_48k.wav")
    source, sample_rate = load_audio(source_pcm)
    vocal_paths: list[Path] = []
    instrumental_paths: list[Path] = []
    separation_report = None

    explicit_vocal = materialize_input(
        payload,
        "vocal",
        job_dir / "explicit_vocal",
        required=False,
    )
    explicit_instrumental = materialize_input(
        payload,
        "instrumental",
        job_dir / "explicit_instrumental",
        required=False,
    )
    if explicit_vocal is not None:
        vocal_paths.append(explicit_vocal)
    if explicit_instrumental is not None:
        instrumental_paths.append(explicit_instrumental)

    for index, spec in enumerate(_stem_specs(payload)):
        name = str(spec.get("role") or spec.get("name") or "").lower()
        path = _materialize_stem(
            spec,
            job_dir / "inputs" / f"stem_{index:02d}",
        )
        if any(token in name for token in ("vocal", "voice", "vox")):
            vocal_paths.append(path)
        elif any(
            token in name
            for token in ("instrumental", "no_vocal", "music", "accompaniment")
        ):
            instrumental_paths.append(path)

    if requested_mode == "quick":
        resolved_mode, fallback_reason = "quick", None
    elif requested_mode == "auto":
        resolved_mode = "surgical" if vocal_paths else "quick"
        fallback_reason = (
            None
            if vocal_paths
            else "No vocal stem was available, so auto mode fell back to Quick Naturalize."
        )
    else:
        resolved_mode, fallback_reason = "surgical", None

    if resolved_mode == "surgical" and not vocal_paths:
        if not bool(payload.get("separate_if_needed", True)):
            return {
                "status": "rejected",
                "reason": "Surgical Naturalize requires a vocal stem or separate_if_needed=true.",
            }
        vocals, instrumental, separation_report = _separate(
            source_pcm,
            job_dir,
            str(payload.get("separation_model") or "htdemucs"),
            int(payload.get("timeout_seconds", 7200)),
        )
        vocal_paths = [vocals]
        instrumental_paths = [instrumental]

    vocal_audio = _combine(
        vocal_paths,
        len(source),
        sample_rate,
        job_dir / "vocal_pcm",
    )
    instrumental_audio = _combine(
        instrumental_paths,
        len(source),
        sample_rate,
        job_dir / "instrumental_pcm",
    )
    processed, report = naturalize_arrays(
        source,
        sample_rate,
        mode=resolved_mode,
        intensity=intensity,
        vocal=vocal_audio,
        instrumental=instrumental_audio,
    )
    report.update(
        {
            "artist": artist,
            "song": song,
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "automatic_mode_selection": requested_mode == "auto",
            "fallback_reason": fallback_reason,
            "separation": separation_report,
            "agent_log": {
                "callable_operation": "naturalize",
                "preferred_mode_when_stems_available": "surgical",
                "fallback_without_stems": "quick",
                "original_retained_for_ab": bool(
                    payload.get("retain_original", True)
                ),
                "parameters": {
                    "intensity": round(intensity, 4),
                    "nominal_cocktail": NOMINAL_COCKTAIL,
                },
                "resulting_characteristics": report[
                    "naturalness_characteristics"
                ],
            },
        }
    )
    if report["status"] != "completed":
        return report

    safe_song = _slug(song)
    naturalized_path = job_dir / f"{safe_song}_Naturalized.wav"
    original_path = job_dir / f"{safe_song}_Original_AB_Reference.wav"
    report_path = job_dir / f"{safe_song}_Naturalize_Report.json"
    write_audio(naturalized_path, processed, sample_rate, allow_clip=False)
    if bool(payload.get("retain_original", True)):
        write_audio(original_path, source, sample_rate, allow_clip=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["processed_path"] = str(naturalized_path)
    report["original_reference_path"] = (
        str(original_path) if original_path.exists() else None
    )
    report["report_path"] = str(report_path)
    if bool(payload.get("publish_outputs", True)):
        outputs = [naturalized_path, report_path]
        if original_path.exists():
            outputs.insert(1, original_path)
        report["outputs"] = publish_files(
            outputs,
            artist=artist,
            category="naturalized",
            ttl_seconds=int(payload.get("output_ttl_seconds", 86400)),
        )
    return report
