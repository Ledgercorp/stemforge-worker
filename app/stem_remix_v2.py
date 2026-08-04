from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path

import numpy as np
from scipy import signal

from app.analysis_v2 import analyze_array
from app.audio_core import (
    ensure_pcm_wav,
    highpass,
    limit_peak,
    linked_compressor,
    load_audio,
    normalize_loudness,
    peak,
    write_audio,
)
from app.full_pass_v2 import _analyze_midi, _materialize_stems, _safe_extract_midi, _stem_specs
from app.mastering_v2 import _upload_github_release_asset
from app.quality_v2 import compare_audio_quality, quality_gate
from app.storage import WORKSPACE, materialize_input, publish_files


ROLE_ALIASES = {
    "lead vocals": "lead_vocals",
    "lead vocal": "lead_vocals",
    "vocals": "lead_vocals",
    "backing vocals": "backing_vocals",
    "backing vocal": "backing_vocals",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "percussion": "percussion",
    "synth": "synth",
    "other": "other",
    "pad": "pad",
    "bass pad": "bass_pad",
    "bass_pad": "bass_pad",
    "breakdown pad": "bass_pad",
    "full mix": "full_mix",
    "full_mix": "full_mix",
}

AUXILIARY_ROLES = {"pad", "bass_pad"}

MIX_PROFILES = {
    "balanced": {
        "target_lufs": -12.8,
        "ceiling_dbfs": -1.1,
        "bus_threshold_db": -13.0,
        "bus_ratio": 1.30,
        "bus_attack_ms": 34.0,
        "role_gain_db": {
            "lead_vocals": 0.8,
            "backing_vocals": 0.5,
            "drums": 0.7,
            "bass": 0.7,
            "guitar": -0.2,
            "percussion": 0.3,
            "synth": 0.2,
            "other": 0.0,
            "pad": -4.5,
            "bass_pad": -4.0,
        },
        "delta_amount": {
            "lead_vocals": 0.32,
            "backing_vocals": 0.24,
            "drums": 0.26,
            "bass": 0.28,
            "guitar": 0.18,
            "percussion": 0.18,
            "synth": 0.18,
            "other": 0.12,
        },
    },
    "heavy": {
        "target_lufs": -11.2,
        "ceiling_dbfs": -1.0,
        "bus_threshold_db": -15.0,
        "bus_ratio": 1.55,
        "bus_attack_ms": 27.0,
        "role_gain_db": {
            "lead_vocals": 0.7,
            "backing_vocals": 0.3,
            "drums": 1.2,
            "bass": 1.2,
            "guitar": 0.2,
            "percussion": 0.4,
            "synth": 0.1,
            "other": -0.1,
            "pad": -3.2,
            "bass_pad": -2.8,
        },
        "delta_amount": {
            "lead_vocals": 0.34,
            "backing_vocals": 0.24,
            "drums": 0.34,
            "bass": 0.36,
            "guitar": 0.22,
            "percussion": 0.20,
            "synth": 0.18,
            "other": 0.12,
        },
    },
}


def _safe_role(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.strip().lower())
    return ROLE_ALIASES.get(lowered, lowered.replace(" ", "_"))


def _gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * (10.0 ** (float(gain_db) / 20.0))).astype(np.float32)


def _sos_filter(
    audio: np.ndarray,
    sample_rate: int,
    kind: str,
    frequency_hz: float,
    order: int = 2,
) -> np.ndarray:
    frequency_hz = float(np.clip(frequency_hz, 10.0, sample_rate * 0.45))
    sos = signal.butter(
        order,
        frequency_hz,
        btype=kind,
        fs=sample_rate,
        output="sos",
    )
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _bell(
    audio: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    gain_db: float,
    q: float = 0.8,
) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * frequency_hz / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    cos_omega = math.cos(omega)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / a
    sos = np.array(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]],
        dtype=np.float64,
    )
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _prepare_role(
    audio: np.ndarray,
    sample_rate: int,
    role: str,
    release_ms: float,
) -> tuple[np.ndarray, list[dict]]:
    """Conservative stem processing without autonomous stereo widening."""
    processed = audio.astype(np.float32)
    chain: list[dict] = []

    if role == "lead_vocals":
        processed = highpass(processed, sample_rate, cutoff_hz=72.0)
        processed = _bell(processed, sample_rate, 300.0, -0.8, 0.9)
        processed = _bell(processed, sample_rate, 2600.0, 0.7, 0.8)
        processed = linked_compressor(
            processed,
            sample_rate,
            threshold_db=-18.0,
            ratio=1.8,
            attack_ms=18.0,
            release_ms=max(120.0, release_ms * 0.75),
        )
        chain = [
            {"type": "highpass", "hz": 72.0},
            {"type": "bell", "hz": 300.0, "gain_db": -0.8},
            {"type": "bell", "hz": 2600.0, "gain_db": 0.7},
            {"type": "compressor", "threshold_db": -18.0, "ratio": 1.8},
        ]
    elif role == "backing_vocals":
        processed = highpass(processed, sample_rate, cutoff_hz=105.0)
        processed = _bell(processed, sample_rate, 350.0, -1.0, 0.85)
        processed = linked_compressor(
            processed,
            sample_rate,
            threshold_db=-24.0,
            ratio=2.0,
            attack_ms=22.0,
            release_ms=max(140.0, release_ms * 0.9),
        )
        chain = [
            {"type": "highpass", "hz": 105.0},
            {"type": "bell", "hz": 350.0, "gain_db": -1.0},
            {"type": "compressor", "threshold_db": -24.0, "ratio": 2.0},
        ]
    elif role == "drums":
        processed = highpass(processed, sample_rate, cutoff_hz=24.0)
        compressed = linked_compressor(
            processed,
            sample_rate,
            threshold_db=-17.0,
            ratio=2.8,
            attack_ms=30.0,
            release_ms=max(100.0, release_ms * 0.6),
        )
        processed = (processed * 0.86 + compressed * 0.14).astype(np.float32)
        processed = _bell(processed, sample_rate, 68.0, 0.4, 0.85)
        chain = [
            {"type": "highpass", "hz": 24.0},
            {"type": "parallel_compressor", "wet": 0.14, "ratio": 2.8},
            {"type": "bell", "hz": 68.0, "gain_db": 0.4},
        ]
    elif role == "bass":
        processed = highpass(processed, sample_rate, cutoff_hz=24.0)
        processed = _sos_filter(processed, sample_rate, "lowpass", 7500.0)
        processed = _bell(processed, sample_rate, 72.0, 0.5, 0.9)
        processed = _bell(processed, sample_rate, 280.0, -0.5, 0.9)
        processed = linked_compressor(
            processed,
            sample_rate,
            threshold_db=-17.0,
            ratio=2.0,
            attack_ms=35.0,
            release_ms=max(145.0, release_ms * 0.85),
        )
        chain = [
            {"type": "highpass", "hz": 24.0},
            {"type": "lowpass", "hz": 7500.0},
            {"type": "bell", "hz": 72.0, "gain_db": 0.5},
            {"type": "bell", "hz": 280.0, "gain_db": -0.5},
            {"type": "compressor", "threshold_db": -17.0, "ratio": 2.0},
        ]
    elif role == "guitar":
        processed = highpass(processed, sample_rate, cutoff_hz=62.0)
        processed = _bell(processed, sample_rate, 340.0, -0.6, 0.85)
        processed = _bell(processed, sample_rate, 1800.0, 0.3, 0.8)
        chain = [
            {"type": "highpass", "hz": 62.0},
            {"type": "bell", "hz": 340.0, "gain_db": -0.6},
            {"type": "bell", "hz": 1800.0, "gain_db": 0.3},
        ]
    elif role == "percussion":
        processed = highpass(processed, sample_rate, cutoff_hz=100.0)
        processed = _bell(processed, sample_rate, 5500.0, 0.5, 0.8)
        chain = [
            {"type": "highpass", "hz": 100.0},
            {"type": "bell", "hz": 5500.0, "gain_db": 0.5},
        ]
    elif role == "synth":
        processed = highpass(processed, sample_rate, cutoff_hz=38.0)
        processed = _bell(processed, sample_rate, 450.0, -0.4, 0.85)
        chain = [
            {"type": "highpass", "hz": 38.0},
            {"type": "bell", "hz": 450.0, "gain_db": -0.4},
        ]
    elif role in AUXILIARY_ROLES:
        processed = highpass(processed, sample_rate, cutoff_hz=24.0)
        processed = _sos_filter(processed, sample_rate, "lowpass", 1800.0)
        processed = linked_compressor(
            processed,
            sample_rate,
            threshold_db=-20.0,
            ratio=1.7,
            attack_ms=40.0,
            release_ms=max(180.0, release_ms),
        )
        chain = [
            {"type": "highpass", "hz": 24.0},
            {"type": "lowpass", "hz": 1800.0},
            {"type": "compressor", "threshold_db": -20.0, "ratio": 1.7},
        ]
    elif role == "full_mix":
        chain = [{"type": "passthrough"}]
    else:
        processed = highpass(processed, sample_rate, cutoff_hz=28.0)
        chain = [{"type": "highpass", "hz": 28.0}]

    if not np.all(np.isfinite(processed)):
        raise ValueError(f"Processing role '{role}' produced non-finite samples.")
    return processed.astype(np.float32), chain


def _tempo_release(midi_report: dict) -> float:
    bpm = midi_report.get("primary_tempo_bpm")
    if not bpm or float(bpm) <= 0:
        return 180.0
    return float(np.clip((60000.0 / float(bpm)) * 0.36, 140.0, 260.0))


def _correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    length = min(len(reference), len(candidate))
    ref = np.mean(reference[:length].astype(np.float64), axis=1)
    can = np.mean(candidate[:length].astype(np.float64), axis=1)
    ref -= np.mean(ref)
    can -= np.mean(can)
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(can)) + 1e-20
    return float(np.dot(ref, can) / denominator)


def _reconstruction_diagnostics(core_stems: list[np.ndarray], master: np.ndarray) -> dict:
    if not core_stems:
        return {
            "stem_count": 0,
            "least_squares_gain": 1.0,
            "null_residual_relative_db": 0.0,
            "waveform_correlation": 0.0,
            "safe_for_full_stem_mix": False,
            "reasons": ["no_core_stems"],
        }

    stem_sum = np.sum(core_stems, axis=0, dtype=np.float64)
    denominator = float(np.sum(np.square(stem_sum)) + 1e-20)
    gain = float(np.sum(master.astype(np.float64) * stem_sum) / denominator)
    matched = stem_sum * gain
    residual = master.astype(np.float64) - matched
    reference_rms = float(np.sqrt(np.mean(np.square(master, dtype=np.float64)) + 1e-20))
    residual_rms = float(np.sqrt(np.mean(np.square(residual)) + 1e-20))
    residual_db = 20.0 * math.log10(max(residual_rms, 1e-12) / max(reference_rms, 1e-12))
    correlation = _correlation(master, matched)
    reasons: list[str] = []
    if residual_db > -18.0:
        reasons.append("null_residual_above_minus_18_db")
    if correlation < 0.97:
        reasons.append("reconstruction_correlation_below_0_97")
    if gain < 0.5 or gain > 1.5:
        reasons.append("reconstruction_gain_out_of_range")
    return {
        "stem_count": len(core_stems),
        "least_squares_gain": round(gain, 7),
        "null_residual_relative_db": round(residual_db, 4),
        "waveform_correlation": round(correlation, 7),
        "safe_for_full_stem_mix": not reasons,
        "reasons": reasons,
    }


def _select_mode(payload: dict, reconstruction: dict) -> dict:
    requested = str(payload.get("remix_mode") or "auto").strip().lower()
    valid = {"auto", "full_stem_mix", "master_anchored_delta"}
    if requested not in valid:
        raise ValueError(f"Unknown remix_mode '{requested}'. Available modes: {sorted(valid)}")

    safe = bool(reconstruction.get("safe_for_full_stem_mix"))
    if requested == "full_stem_mix" and not safe:
        if not bool(payload.get("allow_unsafe_full_stem_mix", False)):
            return {
                "status": "rejected",
                "requested": requested,
                "selected": None,
                "reason": "The supplied stems do not reconstruct the master safely enough for a full stem replacement mix.",
            }
    selected = (
        "full_stem_mix"
        if requested == "full_stem_mix" or (requested == "auto" and safe)
        else "master_anchored_delta"
    )
    return {
        "status": "selected",
        "requested": requested,
        "selected": selected,
        "reason": (
            "Strict reconstruction thresholds passed."
            if selected == "full_stem_mix"
            else "The coherent stereo master is retained while stem-derived fader and processing deltas are applied."
        ),
    }


def _profile_stem_gain(item: dict, profile: dict) -> float:
    role = item["role"]
    configured = float(profile["role_gain_db"].get(role, 0.0))
    override = float(item.get("gain_db", 0.0))
    return configured + override


def _build_premaster(
    prepared: list[dict],
    master: np.ndarray,
    profile_name: str,
    mode: str,
    reconstruction_gain: float,
) -> tuple[np.ndarray, list[dict]]:
    profile = MIX_PROFILES[profile_name]
    gain_report: list[dict] = []

    if mode == "full_stem_mix":
        mixed = np.zeros_like(master, dtype=np.float64)
        for item in prepared:
            role = item["role"]
            gain_db = _profile_stem_gain(item, profile)
            if item["overlay"]:
                total_gain = 10.0 ** (gain_db / 20.0)
            else:
                total_gain = reconstruction_gain * (10.0 ** (gain_db / 20.0))
            mixed += item["processed"].astype(np.float64) * total_gain
            gain_report.append(
                {
                    "name": item["name"],
                    "role": role,
                    "mode": "direct_sum",
                    "gain_db": round(20.0 * math.log10(max(total_gain, 1e-12)), 4),
                }
            )
    else:
        mixed = master.astype(np.float64).copy()
        for item in prepared:
            role = item["role"]
            gain_db = _profile_stem_gain(item, profile)
            if item["overlay"]:
                overlay_gain = 10.0 ** (gain_db / 20.0)
                mixed += item["processed"].astype(np.float64) * overlay_gain
                gain_report.append(
                    {
                        "name": item["name"],
                        "role": role,
                        "mode": "direct_overlay",
                        "gain_db": round(gain_db, 4),
                    }
                )
                continue

            fader_scale = 10.0 ** (gain_db / 20.0) - 1.0
            delta_amount = float(profile["delta_amount"].get(role, 0.12))
            delta_amount *= float(item.get("delta_amount", 1.0))
            fader_delta = item["raw"].astype(np.float64) * fader_scale
            processing_delta = (
                item["processed"].astype(np.float64)
                - item["raw"].astype(np.float64)
            ) * delta_amount
            mixed += fader_delta + processing_delta
            gain_report.append(
                {
                    "name": item["name"],
                    "role": role,
                    "mode": "master_anchored_delta",
                    "fader_gain_db": round(gain_db, 4),
                    "processing_delta_amount": round(delta_amount, 5),
                }
            )

    if not np.all(np.isfinite(mixed)):
        raise ValueError("Stem mix produced non-finite samples.")
    return mixed.astype(np.float32), gain_report


def _peak_trim(audio: np.ndarray, ceiling_dbfs: float) -> tuple[np.ndarray, float]:
    ceiling = 10.0 ** (float(ceiling_dbfs) / 20.0)
    maximum = peak(audio)
    if maximum <= ceiling:
        return audio.astype(np.float32), 0.0
    factor = ceiling / maximum
    return (audio * factor).astype(np.float32), 20.0 * math.log10(factor)


def _finish_profile(
    premaster: np.ndarray,
    sample_rate: int,
    profile_name: str,
    release_ms: float,
) -> tuple[np.ndarray, dict]:
    settings = MIX_PROFILES[profile_name]
    processed = highpass(premaster, sample_rate, cutoff_hz=18.0)
    processed = linked_compressor(
        processed,
        sample_rate,
        threshold_db=float(settings["bus_threshold_db"]),
        ratio=float(settings["bus_ratio"]),
        attack_ms=float(settings["bus_attack_ms"]),
        release_ms=release_ms,
    )
    processed, loudness = normalize_loudness(
        processed,
        sample_rate,
        float(settings["target_lufs"]),
        maximum_gain_db=6.0,
    )
    processed, peak_trim = limit_peak(
        processed,
        ceiling_dbfs=float(settings["ceiling_dbfs"]),
    )
    if not np.all(np.isfinite(processed)):
        raise ValueError(f"Profile '{profile_name}' produced non-finite samples.")
    return processed.astype(np.float32), {
        "profile": profile_name,
        "highpass_hz": 18.0,
        "stereo_width_processing": False,
        "compressor_threshold_db": float(settings["bus_threshold_db"]),
        "compressor_ratio": float(settings["bus_ratio"]),
        "compressor_attack_ms": float(settings["bus_attack_ms"]),
        "compressor_release_ms": round(release_ms, 4),
        **loudness,
        **peak_trim,
    }


def _github_config(payload: dict) -> tuple[str, int, str] | None:
    config = payload.get("github_release_export")
    if not config:
        return None
    repository = str(config.get("repository") or "").strip()
    release_id = int(config.get("release_id") or 0)
    token = str(config.get("token") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("github_release_export.repository must be owner/repository.")
    if release_id <= 0 or not token:
        raise ValueError("Invalid github_release_export configuration.")
    return repository, release_id, token


def stem_remix_job(payload: dict) -> dict:
    artist = str(payload.get("artist") or "sounddecay")
    song = str(payload.get("song") or "untitled")
    requested = payload.get("mix_profiles") or ["balanced", "heavy"]
    if isinstance(requested, str):
        requested = [requested]
    invalid = [name for name in requested if name not in MIX_PROFILES]
    if invalid:
        return {"status": "rejected", "reason": f"Unknown mix profiles: {invalid}"}

    specs = _stem_specs(payload)
    if len(specs) < 1:
        return {"status": "rejected", "reason": "At least one stem is required."}

    job_id = uuid.uuid4().hex[:16]
    job_root = (WORKSPACE / "jobs" / "stem_remix" / job_id).resolve()
    job_root.mkdir(parents=True, exist_ok=True)

    master_source = materialize_input(payload, "master", job_root / "master_source")
    master_pcm = ensure_pcm_wav(master_source, job_root / "master.wav")
    master, sample_rate = load_audio(master_pcm)

    stem_paths = _materialize_stems(specs, job_root)
    loaded: list[dict] = []
    shortest = len(master)
    for index, ((name, path), spec) in enumerate(zip(stem_paths, specs)):
        pcm = ensure_pcm_wav(path, job_root / "pcm" / f"{index:02d}.wav")
        audio, stem_rate = load_audio(pcm)
        if stem_rate != sample_rate:
            raise ValueError(f"Unexpected sample-rate mismatch for {name}: {stem_rate}")
        shortest = min(shortest, len(audio))
        explicit_role = str(spec.get("role") or name)
        role = _safe_role(explicit_role)
        overlay = bool(spec.get("overlay", False)) or role in AUXILIARY_ROLES
        loaded.append(
            {
                "name": name,
                "role": role,
                "overlay": overlay,
                "gain_db": float(spec.get("gain_db", 0.0)),
                "delta_amount": float(spec.get("delta_amount", 1.0)),
                "raw": audio,
            }
        )

    master = master[:shortest]
    for item in loaded:
        item["raw"] = item["raw"][:shortest]

    midi_paths: list[Path] = []
    midi_archive = materialize_input(
        payload,
        "midi_zip",
        job_root / "midi.zip",
        required=False,
    )
    if midi_archive is not None:
        midi_paths = _safe_extract_midi(midi_archive, job_root / "midi")
    midi_report = _analyze_midi(midi_paths)
    release_ms = _tempo_release(midi_report)

    core_raw = [item["raw"] for item in loaded if not item["overlay"]]
    reconstruction = _reconstruction_diagnostics(core_raw, master)
    mode_decision = _select_mode(payload, reconstruction)
    if mode_decision["status"] == "rejected":
        return {
            "status": "rejected",
            "reason": mode_decision["reason"],
            "reconstruction": reconstruction,
            "mode_decision": mode_decision,
        }
    selected_mode = str(mode_decision["selected"])

    prepared: list[dict] = []
    stem_processing: list[dict] = []
    for item in loaded:
        processed, chain = _prepare_role(
            item["raw"],
            sample_rate,
            item["role"],
            release_ms,
        )
        prepared_item = {**item, "processed": processed}
        prepared.append(prepared_item)
        stem_processing.append(
            {
                "name": item["name"],
                "role": item["role"],
                "overlay": item["overlay"],
                "input_metrics": analyze_array(item["raw"], sample_rate),
                "processing": chain,
                "processed_metrics": analyze_array(processed, sample_rate),
            }
        )

    safe_song = re.sub(r"[^A-Za-z0-9._-]+", "_", song).strip("._-") or "untitled"
    outputs: list[Path] = []
    mixes: list[dict] = []

    for profile_name in requested:
        profile_premaster, stem_gains = _build_premaster(
            prepared,
            master,
            profile_name,
            selected_mode,
            float(reconstruction["least_squares_gain"]),
        )
        rendered, bus_chain = _finish_profile(
            profile_premaster,
            sample_rate,
            profile_name,
            release_ms,
        )
        metrics = analyze_array(rendered, sample_rate)
        safety = quality_gate(
            master,
            rendered,
            sample_rate,
            metrics,
            mode=selected_mode,
        )
        path = job_root / f"{safe_song}_Stem_Remix_{profile_name}.wav"
        write_audio(path, rendered, sample_rate)
        outputs.append(path)
        mixes.append(
            {
                "profile": profile_name,
                "mix_mode": selected_mode,
                "stem_gains": stem_gains,
                "bus_chain": bus_chain,
                "metrics": metrics,
                "reference_comparison": compare_audio_quality(master, rendered, sample_rate),
                "local_path": str(path),
                "safety": safety,
            }
        )

    balanced_premaster, premaster_stem_gains = _build_premaster(
        prepared,
        master,
        "balanced",
        selected_mode,
        float(reconstruction["least_squares_gain"]),
    )
    balanced_premaster, premaster_trim_db = _peak_trim(balanced_premaster, -6.0)
    premaster_path = job_root / f"{safe_song}_Stem_Remix_Premaster.wav"
    write_audio(premaster_path, balanced_premaster, sample_rate)
    outputs.append(premaster_path)

    report = {
        "status": "completed",
        "engine": "StemForge coherence-safe stem remix engine v2.2.1",
        "artist": artist,
        "song": song,
        "input_summary": {
            "stem_count": len(prepared),
            "core_stem_count": len(core_raw),
            "auxiliary_stem_count": len(prepared) - len(core_raw),
            "stem_names": [item["name"] for item in prepared],
            "midi_file_count": len(midi_paths),
            "duration_seconds": round(shortest / sample_rate, 6),
            "sample_rate": sample_rate,
        },
        "midi_analysis": midi_report,
        "tempo_linked_release_ms": round(release_ms, 4),
        "reconstruction": reconstruction,
        "mode_decision": mode_decision,
        "reference_policy": {
            "master_used_as_audio_source": selected_mode == "master_anchored_delta",
            "master_used_for": (
                ["coherent mix foundation", "quality reference"]
                if selected_mode == "master_anchored_delta"
                else ["strict reconstruction validation", "quality reference"]
            ),
            "auxiliary_stems_excluded_from_reconstruction": True,
        },
        "stem_processing": stem_processing,
        "mixes": mixes,
        "premaster": {
            "filename": premaster_path.name,
            "mix_mode": selected_mode,
            "stem_gains": premaster_stem_gains,
            "peak_trim_db": round(premaster_trim_db, 4),
            "bus_compression": False,
            "loudness_normalization": False,
            "limiting": False,
            "metrics": analyze_array(balanced_premaster, sample_rate),
        },
        "honesty_note": (
            "A full stem-replacement mix is used only when the supplied stems pass strict reconstruction thresholds. "
            "Otherwise the worker performs a master-anchored delta remix so stem-level treatment does not expose separation leakage, phase cancellation, or static-like artifacts."
        ),
    }

    failed_profiles = [
        item["profile"] for item in mixes if not item["safety"]["accepted"]
    ]
    report_path = job_root / f"{safe_song}_Stem_Remix_Report.json"
    if failed_profiles:
        report["status"] = "rejected"
        report["reason"] = f"Quality gate rejected profiles: {failed_profiles}"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    github = _github_config(payload)
    if github is not None:
        repository, release_id, token = github
        assets: list[dict] = []
        for path in outputs:
            asset = _upload_github_release_asset(
                path,
                repository=repository,
                release_id=release_id,
                token=token,
            )
            assets.append(asset)
            for mix in mixes:
                if Path(mix["local_path"]).name == path.name:
                    mix["output"] = asset
            if path == premaster_path:
                report["premaster"]["output"] = asset

        report["github_release_export"] = {
            "repository": repository,
            "release_id": release_id,
            "assets": assets,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report_asset = _upload_github_release_asset(
            report_path,
            repository=repository,
            release_id=release_id,
            token=token,
        )
        report["report_output"] = report_asset
        report["github_release_export"]["assets"].append(report_asset)
        return report

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs.append(report_path)
    published = publish_files(
        outputs,
        artist=artist,
        category="stem_remixes",
        ttl_seconds=int(payload.get("output_ttl_seconds", 86400)),
    )
    by_name = {item["filename"]: item for item in published}
    for mix in mixes:
        mix["output"] = by_name[Path(mix["local_path"]).name]
    report["premaster"]["output"] = by_name[premaster_path.name]
    report["report_output"] = by_name[report_path.name]
    return report
