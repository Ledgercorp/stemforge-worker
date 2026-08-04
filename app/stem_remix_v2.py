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
    apply_mid_side_width,
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
}

MIX_PROFILES = {
    "balanced": {
        "role_gain_db": {
            "lead_vocals": 1.0,
            "backing_vocals": 5.0,
            "drums": 1.1,
            "bass": 1.0,
            "guitar": -0.4,
            "percussion": 2.5,
            "synth": 0.5,
            "other": 0.0,
        },
        "target_lufs": -12.5,
        "ceiling_dbfs": -1.0,
        "bus_ratio": 1.55,
        "bus_threshold_db": -14.0,
        "bus_attack_ms": 30.0,
        "bus_width": 0.99,
    },
    "heavy": {
        "role_gain_db": {
            "lead_vocals": 0.8,
            "backing_vocals": 4.0,
            "drums": 2.0,
            "bass": 2.0,
            "guitar": 0.5,
            "percussion": 2.0,
            "synth": 0.3,
            "other": -0.3,
        },
        "target_lufs": -10.8,
        "ceiling_dbfs": -0.9,
        "bus_ratio": 2.0,
        "bus_threshold_db": -16.0,
        "bus_attack_ms": 22.0,
        "bus_width": 0.98,
    },
}


def _safe_role(name: str) -> str:
    lowered = re.sub(r"\s+", " ", name.strip().lower())
    return ROLE_ALIASES.get(lowered, lowered.replace(" ", "_"))


def _gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * (10.0 ** (float(gain_db) / 20.0))).astype(np.float32)


def _sos_filter(audio: np.ndarray, sample_rate: int, kind: str, frequency_hz: float, order: int = 2) -> np.ndarray:
    frequency_hz = float(np.clip(frequency_hz, 10.0, sample_rate * 0.45))
    sos = signal.butter(order, frequency_hz, btype=kind, fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _bell(audio: np.ndarray, sample_rate: int, frequency_hz: float, gain_db: float, q: float = 0.8) -> np.ndarray:
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
    sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _prepare_role(audio: np.ndarray, sample_rate: int, role: str, release_ms: float) -> tuple[np.ndarray, list[dict]]:
    processed = audio.astype(np.float32)
    chain: list[dict] = []

    if role == "lead_vocals":
        processed = highpass(processed, sample_rate, cutoff_hz=75.0)
        processed = _bell(processed, sample_rate, 280.0, -1.2, 0.9)
        processed = _bell(processed, sample_rate, 2600.0, 1.0, 0.75)
        processed = linked_compressor(processed, sample_rate, threshold_db=-19.0, ratio=2.2, attack_ms=14.0, release_ms=max(110.0, release_ms * 0.75))
        processed = apply_mid_side_width(processed, 0.72)
        chain = [
            {"type": "highpass", "hz": 75.0},
            {"type": "bell", "hz": 280.0, "gain_db": -1.2},
            {"type": "bell", "hz": 2600.0, "gain_db": 1.0},
            {"type": "compressor", "threshold_db": -19.0, "ratio": 2.2},
            {"type": "width", "amount": 0.72},
        ]
    elif role == "backing_vocals":
        processed = highpass(processed, sample_rate, cutoff_hz=115.0)
        processed = _bell(processed, sample_rate, 350.0, -1.5, 0.8)
        processed = linked_compressor(processed, sample_rate, threshold_db=-27.0, ratio=2.6, attack_ms=20.0, release_ms=max(130.0, release_ms * 0.85))
        processed = apply_mid_side_width(processed, 1.28)
        chain = [
            {"type": "highpass", "hz": 115.0},
            {"type": "bell", "hz": 350.0, "gain_db": -1.5},
            {"type": "compressor", "threshold_db": -27.0, "ratio": 2.6},
            {"type": "width", "amount": 1.28},
        ]
    elif role == "drums":
        processed = highpass(processed, sample_rate, cutoff_hz=25.0)
        compressed = linked_compressor(processed, sample_rate, threshold_db=-18.0, ratio=4.0, attack_ms=28.0, release_ms=max(90.0, release_ms * 0.55))
        processed = (processed * 0.78 + compressed * 0.22).astype(np.float32)
        processed = _bell(processed, sample_rate, 65.0, 0.7, 0.8)
        processed = _bell(processed, sample_rate, 4200.0, 0.7, 0.7)
        chain = [
            {"type": "highpass", "hz": 25.0},
            {"type": "parallel_compressor", "wet": 0.22, "ratio": 4.0},
            {"type": "bell", "hz": 65.0, "gain_db": 0.7},
            {"type": "bell", "hz": 4200.0, "gain_db": 0.7},
        ]
    elif role == "bass":
        processed = highpass(processed, sample_rate, cutoff_hz=25.0)
        processed = _sos_filter(processed, sample_rate, "lowpass", 9000.0)
        processed = _bell(processed, sample_rate, 72.0, 0.8, 0.85)
        processed = _bell(processed, sample_rate, 280.0, -0.8, 0.9)
        processed = linked_compressor(processed, sample_rate, threshold_db=-18.0, ratio=2.8, attack_ms=32.0, release_ms=max(130.0, release_ms * 0.8))
        processed = apply_mid_side_width(processed, 0.35)
        chain = [
            {"type": "highpass", "hz": 25.0},
            {"type": "lowpass", "hz": 9000.0},
            {"type": "bell", "hz": 72.0, "gain_db": 0.8},
            {"type": "bell", "hz": 280.0, "gain_db": -0.8},
            {"type": "compressor", "threshold_db": -18.0, "ratio": 2.8},
            {"type": "width", "amount": 0.35},
        ]
    elif role == "guitar":
        processed = highpass(processed, sample_rate, cutoff_hz=68.0)
        processed = _bell(processed, sample_rate, 330.0, -0.8, 0.85)
        processed = _bell(processed, sample_rate, 1800.0, 0.5, 0.75)
        processed = apply_mid_side_width(processed, 1.08)
        chain = [
            {"type": "highpass", "hz": 68.0},
            {"type": "bell", "hz": 330.0, "gain_db": -0.8},
            {"type": "bell", "hz": 1800.0, "gain_db": 0.5},
            {"type": "width", "amount": 1.08},
        ]
    elif role == "percussion":
        processed = highpass(processed, sample_rate, cutoff_hz=110.0)
        processed = _bell(processed, sample_rate, 5500.0, 1.0, 0.75)
        processed = apply_mid_side_width(processed, 1.18)
        chain = [
            {"type": "highpass", "hz": 110.0},
            {"type": "bell", "hz": 5500.0, "gain_db": 1.0},
            {"type": "width", "amount": 1.18},
        ]
    elif role == "synth":
        processed = highpass(processed, sample_rate, cutoff_hz=42.0)
        processed = _bell(processed, sample_rate, 450.0, -0.7, 0.8)
        processed = apply_mid_side_width(processed, 1.12)
        chain = [
            {"type": "highpass", "hz": 42.0},
            {"type": "bell", "hz": 450.0, "gain_db": -0.7},
            {"type": "width", "amount": 1.12},
        ]
    else:
        processed = highpass(processed, sample_rate, cutoff_hz=30.0)
        processed = apply_mid_side_width(processed, 1.03)
        chain = [{"type": "highpass", "hz": 30.0}, {"type": "width", "amount": 1.03}]

    return processed.astype(np.float32), chain


def _envelope(audio: np.ndarray, sample_rate: int, window_seconds: float = 0.25) -> np.ndarray:
    mono = np.mean(audio, axis=1, dtype=np.float64)
    window = max(1, int(sample_rate * window_seconds))
    count = len(mono) // window
    if count <= 0:
        return np.array([float(np.sqrt(np.mean(mono**2) + 1e-20))])
    framed = mono[: count * window].reshape(count, window)
    return np.sqrt(np.mean(framed**2, axis=1) + 1e-20)


def _reference_gains(stems: list[np.ndarray], master: np.ndarray, sample_rate: int) -> np.ndarray:
    x = np.column_stack([_envelope(item, sample_rate) for item in stems])
    y = _envelope(master, sample_rate)
    length = min(len(y), len(x))
    x = x[:length]
    y = y[:length]
    scale = max(float(np.trace(x.T @ x) / max(1, x.shape[1])), 1e-12)
    regularization = scale * 0.18
    matrix = x.T @ x + np.eye(x.shape[1]) * regularization
    target = x.T @ y + np.ones(x.shape[1]) * regularization
    try:
        gains = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        gains = np.ones(x.shape[1], dtype=np.float64)
    return np.clip(gains, 0.55, 1.65)


def _tempo_release(midi_report: dict) -> float:
    bpm = midi_report.get("primary_tempo_bpm")
    if not bpm or float(bpm) <= 0:
        return 180.0
    return float(np.clip((60000.0 / float(bpm)) * 0.36, 140.0, 260.0))


def _mix_profile(prepared: list[dict], reference_gains: np.ndarray, sample_rate: int, profile_name: str, release_ms: float) -> tuple[np.ndarray, dict]:
    settings = MIX_PROFILES[profile_name]
    mixed = np.zeros_like(prepared[0]["audio"], dtype=np.float64)
    gain_report: list[dict] = []

    for index, item in enumerate(prepared):
        role = item["role"]
        creative_db = float(settings["role_gain_db"].get(role, 0.0))
        reference_gain = float(reference_gains[index])
        total_gain = reference_gain * (10.0 ** (creative_db / 20.0))
        mixed += item["audio"].astype(np.float64) * total_gain
        gain_report.append({
            "name": item["name"],
            "role": role,
            "reference_gain_linear": round(reference_gain, 6),
            "reference_gain_db": round(20.0 * math.log10(max(reference_gain, 1e-12)), 3),
            "creative_offset_db": creative_db,
            "total_gain_db": round(20.0 * math.log10(max(total_gain, 1e-12)), 3),
        })

    mixed = mixed.astype(np.float32)
    maximum = peak(mixed)
    headroom_ceiling = 10.0 ** (-6.0 / 20.0)
    headroom_gain_db = 0.0
    if maximum > headroom_ceiling:
        factor = headroom_ceiling / maximum
        mixed *= factor
        headroom_gain_db = 20.0 * math.log10(factor)

    mixed = highpass(mixed, sample_rate, cutoff_hz=20.0)
    mixed = _bell(mixed, sample_rate, 115.0, -0.6 if profile_name == "balanced" else -0.3, 0.75)
    mixed = _bell(mixed, sample_rate, 1900.0, 0.5, 0.7)
    mixed = apply_mid_side_width(mixed, float(settings["bus_width"]))
    mixed = linked_compressor(
        mixed,
        sample_rate,
        threshold_db=float(settings["bus_threshold_db"]),
        ratio=float(settings["bus_ratio"]),
        attack_ms=float(settings["bus_attack_ms"]),
        release_ms=release_ms,
    )
    mixed, loudness = normalize_loudness(mixed, sample_rate, float(settings["target_lufs"]), maximum_gain_db=10.0)
    mixed, limiter = limit_peak(mixed, ceiling_dbfs=float(settings["ceiling_dbfs"]))

    return mixed.astype(np.float32), {
        "profile": profile_name,
        "stem_gains": gain_report,
        "headroom_gain_db": round(headroom_gain_db, 3),
        "bus": {
            "highpass_hz": 20.0,
            "low_bell_db": -0.6 if profile_name == "balanced" else -0.3,
            "presence_bell_db": 0.5,
            "width": float(settings["bus_width"]),
            "compressor_threshold_db": float(settings["bus_threshold_db"]),
            "compressor_ratio": float(settings["bus_ratio"]),
            "compressor_attack_ms": float(settings["bus_attack_ms"]),
            "compressor_release_ms": round(release_ms, 3),
            **loudness,
            **limiter,
        },
    }


def _reference_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict:
    length = min(len(reference), len(candidate))
    ref = reference[:length].astype(np.float64)
    can = candidate[:length].astype(np.float64)
    ref_mono = np.mean(ref, axis=1)
    can_mono = np.mean(can, axis=1)
    ref_mono -= ref_mono.mean()
    can_mono -= can_mono.mean()
    denominator = np.sqrt(np.dot(ref_mono, ref_mono) * np.dot(can_mono, can_mono)) + 1e-20
    correlation = float(np.dot(ref_mono, can_mono) / denominator)
    residual = ref - can
    residual_relative_db = 20.0 * math.log10(
        max(float(np.sqrt(np.mean(residual**2) + 1e-20)), 1e-12)
        / max(float(np.sqrt(np.mean(ref**2) + 1e-20)), 1e-12)
    )
    return {
        "waveform_correlation": round(correlation, 6),
        "residual_relative_db": round(residual_relative_db, 3),
        "note": "The supplied master was used only as a diagnostic balance reference and was not mixed into the remix.",
    }


def _github_config(payload: dict) -> tuple[str, int, str] | None:
    config = payload.get("github_release_export")
    if not config:
        return None
    repository = str(config.get("repository") or "").strip()
    release_id = int(config.get("release_id") or 0)
    token = str(config.get("token") or "").strip()
    if "/" not in repository or release_id <= 0 or not token:
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
    if len(specs) < 2:
        return {"status": "rejected", "reason": "At least two stems are required."}

    job_id = uuid.uuid4().hex[:16]
    job_root = (WORKSPACE / "jobs" / "stem_remix" / job_id).resolve()
    job_root.mkdir(parents=True, exist_ok=True)

    master_source = materialize_input(payload, "master", job_root / "master_source")
    master_pcm = ensure_pcm_wav(master_source, job_root / "master.wav")
    master, sample_rate = load_audio(master_pcm)

    stem_paths = _materialize_stems(specs, job_root)
    loaded: list[dict] = []
    raw_stems: list[np.ndarray] = []
    shortest = len(master)
    for name, path in stem_paths:
        pcm = ensure_pcm_wav(path, job_root / "pcm" / f"{len(loaded):02d}.wav")
        audio, stem_rate = load_audio(pcm)
        if stem_rate != sample_rate:
            raise ValueError(f"Unexpected sample-rate mismatch for {name}: {stem_rate}")
        shortest = min(shortest, len(audio))
        raw_stems.append(audio)
        loaded.append({"name": name, "role": _safe_role(name)})

    master = master[:shortest]
    raw_stems = [item[:shortest] for item in raw_stems]

    midi_paths: list[Path] = []
    midi_archive = materialize_input(payload, "midi_zip", job_root / "midi.zip", required=False)
    if midi_archive is not None:
        midi_paths = _safe_extract_midi(midi_archive, job_root / "midi")
    midi_report = _analyze_midi(midi_paths)
    release_ms = _tempo_release(midi_report)

    reference_gains = _reference_gains(raw_stems, master, sample_rate)
    prepared: list[dict] = []
    stem_processing: list[dict] = []
    for metadata, audio in zip(loaded, raw_stems):
        processed, chain = _prepare_role(audio, sample_rate, metadata["role"], release_ms)
        prepared.append({**metadata, "audio": processed})
        stem_processing.append({
            "name": metadata["name"],
            "role": metadata["role"],
            "input_metrics": analyze_array(audio, sample_rate),
            "processing": chain,
            "processed_metrics": analyze_array(processed, sample_rate),
        })

    safe_song = re.sub(r"[^A-Za-z0-9._-]+", "_", song).strip("._-") or "untitled"
    outputs: list[Path] = []
    mixes: list[dict] = []

    for profile_name in requested:
        rendered, chain = _mix_profile(prepared, reference_gains, sample_rate, profile_name, release_ms)
        path = job_root / f"{safe_song}_Stem_Remix_{profile_name}.wav"
        write_audio(path, rendered, sample_rate)
        outputs.append(path)
        metrics = analyze_array(rendered, sample_rate)
        reasons = []
        if metrics.get("clipped_sample_count", 0) > 0:
            reasons.append("clipped_samples")
        if float(metrics.get("true_peak_dbtp", 0.0)) > -0.75:
            reasons.append("true_peak_margin")
        mixes.append({
            "profile": profile_name,
            "chain": chain,
            "metrics": metrics,
            "reference_comparison": _reference_comparison(master, rendered),
            "local_path": str(path),
            "safety": {"accepted": not reasons, "reasons": reasons},
        })

    premaster, premaster_chain = _mix_profile(prepared, reference_gains, sample_rate, "balanced", release_ms)
    premaster, premaster_loudness = normalize_loudness(premaster, sample_rate, -16.0, maximum_gain_db=6.0)
    premaster, premaster_limit = limit_peak(premaster, ceiling_dbfs=-3.0)
    premaster_path = job_root / f"{safe_song}_Stem_Remix_Premaster.wav"
    write_audio(premaster_path, premaster, sample_rate)
    outputs.append(premaster_path)

    report = {
        "status": "completed",
        "engine": "StemForge reference-guided stem remix engine v2.2",
        "artist": artist,
        "song": song,
        "input_summary": {
            "stem_count": len(prepared),
            "stem_names": [item["name"] for item in prepared],
            "midi_file_count": len(midi_paths),
            "duration_seconds": round(shortest / sample_rate, 6),
            "sample_rate": sample_rate,
        },
        "reference_policy": {
            "master_used_as_audio_source": False,
            "master_used_for": ["broadband envelope balance fitting", "post-render diagnostic comparison"],
            "timeline_shift_policy": "No automatic stem shifts were applied because the supplied stems share the exact master duration and export origin.",
        },
        "midi_analysis": midi_report,
        "tempo_linked_release_ms": round(release_ms, 3),
        "stem_processing": stem_processing,
        "mixes": mixes,
        "premaster": {
            "filename": premaster_path.name,
            "chain": premaster_chain,
            "final_level": {**premaster_loudness, **premaster_limit},
            "metrics": analyze_array(premaster, sample_rate),
        },
        "honesty_note": (
            "This is a new mix rendered from the supplied stems. The original stereo master was used only as a balance reference and was not blended into the output. "
            "Because the raw stem sum did not null closely against the supplied master, the remix may intentionally differ in bus effects and nonlinear processing."
        ),
    }

    report_path = job_root / f"{safe_song}_Stem_Remix_Report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs.append(report_path)

    github = _github_config(payload)
    if github is not None:
        repository, release_id, token = github
        assets = []
        for path in outputs:
            asset = _upload_github_release_asset(path, repository=repository, release_id=release_id, token=token)
            assets.append(asset)
            for mix in mixes:
                if Path(mix["local_path"]).name == path.name:
                    mix["output"] = asset
            if path == premaster_path:
                report["premaster"]["output"] = asset
            if path == report_path:
                report["report_output"] = asset
        report["github_release_export"] = {"repository": repository, "release_id": release_id, "assets": assets}
        return report

    published = publish_files(outputs, artist=artist, category="stem_remixes", ttl_seconds=int(payload.get("output_ttl_seconds", 86400)))
    by_name = {item["filename"]: item for item in published}
    for mix in mixes:
        mix["output"] = by_name[Path(mix["local_path"]).name]
    report["premaster"]["output"] = by_name[premaster_path.name]
    report["report_output"] = by_name[report_path.name]
    return report
