from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from app.analysis_v2 import analyze_array
from app.audio_core import ensure_pcm_wav, load_audio, write_audio
from app.naturalize_dsp_v2 import (
    NATURALIZE_DSP_VERSION,
    NATURALIZE_LABEL,
    NATURALIZE_PRESET,
    NOMINAL_COCKTAIL,
    PARAMETER_RANGES,
    align_audio,
    apply_headroom,
    artifact_assessment,
    match_level,
    processing_order,
    reconstruction_report,
    render_dsp_role,
    resolve_intensity,
    resolve_parameters,
    resolve_passes,
    role_from_name,
    slug,
)
from app.quality_v2 import compare_audio_quality
from app.storage import materialize_input, publish_files
from app.vocoder_v2 import apply_vocoder, resolve_vocoder

NATURALIZE_VERSION = NATURALIZE_DSP_VERSION

# Compatibility exports used by existing clients and tests.
_resolve_intensity = resolve_intensity
_resolve_parameters = resolve_parameters
_resolve_passes = resolve_passes


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


def _load_stem(
    path: Path,
    length: int,
    sample_rate: int,
    destination: Path,
) -> np.ndarray:
    pcm = ensure_pcm_wav(path, destination, sample_rate=sample_rate)
    audio, _ = load_audio(pcm)
    return align_audio(audio, length)


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


def _prepare_groups(
    source: np.ndarray,
    *,
    vocal: np.ndarray | None,
    instrumental: np.ndarray | None,
    stem_groups: list[dict] | None,
) -> tuple[list[dict], dict | None]:
    groups: list[dict] = []
    for index, item in enumerate(stem_groups or []):
        audio = item.get("audio")
        if audio is None:
            continue
        name = str(item.get("name") or f"stem_{index + 1}")
        role = str(item.get("role") or role_from_name(name))
        groups.append(
            {
                "name": name,
                "role": role,
                "audio": align_audio(audio, len(source)),
            }
        )
    if groups or vocal is None:
        return groups, None

    vocal_audio = align_audio(vocal, len(source))
    groups.append({"name": "vocal", "role": "vocal", "audio": vocal_audio})
    if instrumental is None:
        groups.append(
            {
                "name": "master_minus_vocal",
                "role": "harmonic",
                "audio": (source - vocal_audio).astype(np.float32),
            }
        )
        return groups, {
            "instrumental_source": "master_minus_vocal",
            "reason": "No instrumental stem was supplied.",
        }
    groups.append(
        {
            "name": "instrumental",
            "role": "harmonic",
            "audio": align_audio(instrumental, len(source)),
        }
    )
    return groups, None


def _render_once(
    source: np.ndarray,
    sample_rate: int,
    *,
    mode: str,
    intensity: float,
    vocal: np.ndarray | None,
    instrumental: np.ndarray | None,
    stem_groups: list[dict] | None,
    passes: int,
    second_scale: float,
    parameters: dict,
    vocoder: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if mode == "quick":
        dsp, pre_denoise, role_report = render_dsp_role(
            source,
            sample_rate,
            "full_mix",
            intensity,
            passes,
            second_scale,
            parameters,
            seed,
        )
        quick_vocoder = vocoder
        if vocoder.get("requested") in ("auto", True):
            quick_vocoder = {**vocoder, "enabled": False, "backend": None}
        processed, vocoder_report = apply_vocoder(
            dsp,
            sample_rate,
            quick_vocoder,
            "full_mix",
            intensity,
            seed + 701,
        )
        role_report["vocoder"] = vocoder_report
        return processed, pre_denoise, {
            "mode": "quick",
            "groups": [{"name": "full_mix", **role_report}],
            "reconstruction_gate": None,
            "anchor_decision": {
                "mix_source": "full_mix",
                "reason": "Quick mode processes the supplied complete mix.",
            },
        }

    groups, initial_anchor = _prepare_groups(
        source,
        vocal=vocal,
        instrumental=instrumental,
        stem_groups=stem_groups,
    )
    if not any(item["role"] == "vocal" for item in groups):
        raise ValueError("Surgical Naturalize requires at least one vocal stem.")

    reconstruction = reconstruction_report(source, groups)
    raw_sum = np.sum([item["audio"] for item in groups], axis=0, dtype=np.float64)
    processed_groups = []
    pre_groups = []
    reports = []
    for index, group in enumerate(groups):
        dsp, pre_denoise, role_report = render_dsp_role(
            group["audio"],
            sample_rate,
            group["role"],
            intensity,
            passes,
            second_scale,
            parameters,
            seed + index * 1009,
        )
        processed, vocoder_report = apply_vocoder(
            dsp,
            sample_rate,
            vocoder,
            group["role"],
            intensity,
            seed + index * 1009 + 701,
        )
        role_report["vocoder"] = vocoder_report
        processed_groups.append(processed)
        pre_groups.append(pre_denoise)
        reports.append({"name": group["name"], **role_report})

    processed_sum = np.sum(processed_groups, axis=0, dtype=np.float64)
    pre_sum = np.sum(pre_groups, axis=0, dtype=np.float64)
    if reconstruction["accepted"]:
        processed = processed_sum
        pre_mix = pre_sum
        anchor = initial_anchor or {
            "mix_source": "reconstructed_stem_sum",
            "instrumental_source": "supplied_stems",
            "reason": "Supplied stems passed the strict reconstruction gate.",
        }
    else:
        delta_amount = 0.70
        processed = source.astype(np.float64) + delta_amount * (processed_sum - raw_sum)
        pre_mix = source.astype(np.float64) + delta_amount * (pre_sum - raw_sum)
        anchor = {
            "mix_source": "master_anchored_stem_delta",
            "instrumental_source": (initial_anchor or {}).get(
                "instrumental_source",
                "supplied_stems",
            ),
            "reason": (
                "Supplied stems failed reconstruction, so Naturalize applied "
                "their processing deltas around the coherent master."
            ),
            "delta_amount": delta_amount,
        }

    return (
        processed.astype(np.float32),
        pre_mix.astype(np.float32),
        {
            "mode": "surgical",
            "groups": reports,
            "reconstruction_gate": reconstruction,
            "anchor_decision": anchor,
        },
    )


def _candidate_intensities(intensity: float) -> list[float]:
    if intensity == 0:
        return [0.0]
    output = [intensity]
    for factor in (0.82, 0.68):
        reduced = max(0.6, intensity * factor)
        if all(abs(reduced - value) > 1e-6 for value in output):
            output.append(reduced)
    return output


def _zero_report(mode: str) -> dict:
    return {
        "mode": mode,
        "groups": [],
        "reconstruction_gate": None,
        "anchor_decision": {
            "mix_source": "original",
            "reason": "Explicit zero-intensity bypass.",
        },
    }


def _naturalize_arrays_internal(
    source: np.ndarray,
    sample_rate: int,
    *,
    mode: str,
    intensity: float,
    vocal: np.ndarray | None,
    instrumental: np.ndarray | None,
    stem_groups: list[dict] | None,
    passes: int,
    second_scale: float,
    parameters: dict,
    vocoder: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    attempts = []
    selected = source.copy()
    selected_pre = source.copy()
    selected_render = _zero_report(mode)
    selected_safety = artifact_assessment(
        source,
        source,
        sample_rate,
        0.0,
        stage="dsp",
    )

    for attempt_index, candidate_intensity in enumerate(
        _candidate_intensities(intensity)
    ):
        if candidate_intensity == 0:
            rendered = source.copy()
            pre_denoise = source.copy()
            render_report = _zero_report(mode)
        else:
            rendered, pre_denoise, render_report = _render_once(
                source,
                sample_rate,
                mode=mode,
                intensity=candidate_intensity,
                vocal=vocal,
                instrumental=instrumental,
                stem_groups=stem_groups,
                passes=passes,
                second_scale=second_scale,
                parameters=parameters,
                vocoder=vocoder,
                seed=seed + attempt_index * 7919,
            )
            rendered, level = match_level(source, rendered, 0.75)
            rendered, headroom = apply_headroom(rendered)
            render_report["finalization"] = [level, headroom]

        safety = artifact_assessment(
            source,
            rendered,
            sample_rate,
            candidate_intensity,
            stage="dsp",
        )
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "intensity": round(candidate_intensity, 4),
                "accepted": safety["accepted"],
                "reasons": safety["reasons"],
            }
        )
        selected = rendered
        selected_pre = pre_denoise
        selected_render = render_report
        selected_safety = safety
        if safety["accepted"]:
            break

    selected_intensity = float(attempts[-1]["intensity"])
    before = analyze_array(source, sample_rate)
    after = analyze_array(selected, sample_rate)
    comparison = compare_audio_quality(source, selected, sample_rate)
    operations = [
        operation
        for group in selected_render.get("groups", [])
        for operation in group.get("operations", [])
    ]
    if selected_intensity == 0:
        operations = [
            {"operation": "vibrato", "stage": "primary", "bypassed": True},
            {"operation": "tremolo", "bypassed": True},
            {"operation": "vibrato", "stage": "secondary", "bypassed": True},
            {"operation": "flanger", "bypassed": True},
            {
                "operation": "constrained_post_cocktail_denoise",
                "bypassed": True,
                "maximum_reduction_db": 0.0,
                "exact_mean_reduction_db": 0.0,
            },
        ]

    report = {
        "status": "completed" if selected_safety["accepted"] else "rejected",
        "engine": f"StemForge Naturalize pipeline v{NATURALIZE_VERSION}",
        "preset": NATURALIZE_PRESET,
        "label": NATURALIZE_LABEL,
        "mode": mode,
        "requested_intensity": round(intensity, 4),
        "intensity": round(selected_intensity, 4),
        "intensity_was_reduced": selected_intensity < intensity,
        "non_destructive": True,
        "retain_original": True,
        "retain_pre_denoise": True,
        "processing_order": processing_order(),
        "nominal_cocktail": NOMINAL_COCKTAIL,
        "parameter_ranges": PARAMETER_RANGES,
        "parameters": parameters,
        "passes": {
            "requested": passes,
            "applied": passes,
            "maximum": 2,
            "second_pass_scale": round(second_scale, 4),
            "second_pass_maximum_scale": 0.70,
        },
        "attempts": attempts,
        "operations": operations,
        "stem_routing": selected_render.get("groups", []),
        "reconstruction_gate": selected_render.get("reconstruction_gate"),
        "anchor_decision": selected_render.get("anchor_decision"),
        "vocoder": {
            "request": vocoder,
            "routing": [
                {
                    "name": group.get("name"),
                    "role": group.get("role"),
                    "vocoder": group.get("vocoder"),
                }
                for group in selected_render.get("groups", [])
            ],
        },
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
                float(after["crest_factor_db"])
                - float(before["crest_factor_db"]),
                4,
            ),
            "envelope": selected_safety.get("envelope"),
            "spectral_cosine_similarity": selected_safety.get(
                "spectral_cosine_similarity"
            ),
        },
        "safety": selected_safety,
        "technical_priority": [
            "restore_natural_micro_variations",
            "preserve_them_with_constrained_post_cocktail_denoise",
            "apply_optional_noise_timing_saturation_and_dynamics_layers",
            "optionally_resynthesize_after_denoise",
            "never_sacrifice_musical_transparency",
        ],
        "honesty_note": (
            "Naturalize is a fidelity and authenticity enhancement. It does not "
            "identify, conceal, or guarantee the origin of audio and is not a "
            "detection-evasion tool."
        ),
    }
    return selected.astype(np.float32), selected_pre.astype(np.float32), report


def naturalize_arrays(
    source: np.ndarray,
    sample_rate: int,
    *,
    mode: str = "quick",
    intensity: float = 1.0,
    vocal: np.ndarray | None = None,
    instrumental: np.ndarray | None = None,
    stem_groups: list[dict] | None = None,
    passes: int = 1,
    second_pass_scale: float = 0.65,
    parameters: dict | None = None,
    vocoder: dict | str | bool | None = False,
    seed: int = 1337,
) -> tuple[np.ndarray, dict]:
    mode = str(mode).lower()
    if mode not in {"quick", "surgical"}:
        raise ValueError("Naturalize mode must be quick or surgical.")
    intensity = 0.0 if intensity == 0 else float(np.clip(intensity, 0.6, 1.4))
    passes = max(1, min(int(passes), 2))
    second_pass_scale = float(np.clip(second_pass_scale, 0, 0.70))
    parameter_payload = resolve_parameters(parameters or {})
    vocoder_payload = resolve_vocoder({"vocoder": vocoder}, mode)
    output, _, report = _naturalize_arrays_internal(
        align_audio(source, len(source)),
        sample_rate,
        mode=mode,
        intensity=intensity,
        vocal=vocal,
        instrumental=instrumental,
        stem_groups=stem_groups,
        passes=passes,
        second_scale=second_pass_scale,
        parameters=parameter_payload,
        vocoder=vocoder_payload,
        seed=seed,
    )
    return output, report


def naturalize_audio_job(payload: dict) -> dict:
    artist = str(payload.get("artist") or "sounddecay")
    song = str(payload.get("song") or "untitled")
    requested_mode = str(payload.get("mode") or "auto").lower()
    if requested_mode not in {"auto", "quick", "surgical"}:
        return {"status": "rejected", "reason": "mode must be auto, quick, or surgical."}

    try:
        intensity = resolve_intensity(payload)
        passes, second_scale = resolve_passes(payload)
        parameters = resolve_parameters(payload)
    except (TypeError, ValueError) as exc:
        return {"status": "rejected", "reason": str(exc)}

    job_dir = Path(payload.get("job_dir") or "/tmp/stemforge_naturalize_v2")
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)

    source_input = materialize_input(payload, "audio", job_dir / "source_input")
    source_pcm = ensure_pcm_wav(source_input, job_dir / "source_48k.wav")
    source, sample_rate = load_audio(source_pcm)
    groups = []
    separation_report = None

    for index, spec in enumerate(_stem_specs(payload)):
        path = _materialize_stem(spec, job_dir / "inputs" / f"stem_{index:02d}")
        name = str(spec.get("name") or spec.get("role") or f"stem_{index + 1}")
        role = str(spec.get("role") or role_from_name(name))
        groups.append(
            {
                "name": name,
                "role": role,
                "audio": _load_stem(
                    path,
                    len(source),
                    sample_rate,
                    job_dir / "stem_pcm" / f"{index:02d}_{slug(name)}.wav",
                ),
            }
        )

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
        groups.append(
            {
                "name": "explicit_vocal",
                "role": "vocal",
                "audio": _load_stem(
                    explicit_vocal,
                    len(source),
                    sample_rate,
                    job_dir / "stem_pcm" / "explicit_vocal.wav",
                ),
            }
        )
    if explicit_instrumental is not None:
        groups.append(
            {
                "name": "explicit_instrumental",
                "role": "harmonic",
                "audio": _load_stem(
                    explicit_instrumental,
                    len(source),
                    sample_rate,
                    job_dir / "stem_pcm" / "explicit_instrumental.wav",
                ),
            }
        )

    vocal_available = any(group["role"] == "vocal" for group in groups)
    if requested_mode == "quick":
        resolved_mode, fallback_reason = "quick", None
    elif requested_mode == "auto":
        resolved_mode = "surgical" if vocal_available else "quick"
        fallback_reason = (
            None
            if vocal_available
            else "No vocal stem was available, so auto mode fell back to Quick Naturalize."
        )
    else:
        resolved_mode, fallback_reason = "surgical", None

    if resolved_mode == "surgical" and not vocal_available:
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
        groups = [
            {
                "name": "Demucs Vocals",
                "role": "vocal",
                "audio": _load_stem(
                    vocals,
                    len(source),
                    sample_rate,
                    job_dir / "stem_pcm" / "demucs_vocals.wav",
                ),
            },
            {
                "name": "Demucs Instrumental",
                "role": "harmonic",
                "audio": _load_stem(
                    instrumental,
                    len(source),
                    sample_rate,
                    job_dir / "stem_pcm" / "demucs_instrumental.wav",
                ),
            },
        ]

    try:
        vocoder = resolve_vocoder(payload, resolved_mode)
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc)}

    processed, pre_denoise, report = _naturalize_arrays_internal(
        source,
        sample_rate,
        mode=resolved_mode,
        intensity=intensity,
        vocal=None,
        instrumental=None,
        stem_groups=groups,
        passes=passes,
        second_scale=second_scale,
        parameters=parameters,
        vocoder=vocoder,
        seed=int(payload.get("seed", 1337)),
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
                "preset": NATURALIZE_PRESET,
                "preferred_mode_when_stems_available": "surgical",
                "fallback_without_stems": "quick",
                "original_retained_for_ab": bool(
                    payload.get("retain_original", True)
                ),
                "pre_denoise_retained_for_ab": bool(
                    payload.get("retain_pre_denoise", True)
                ),
                "parameters": {
                    "requested_intensity": round(intensity, 4),
                    "selected_intensity": report["intensity"],
                    "parameters": parameters,
                    "passes": report["passes"],
                    "vocoder": vocoder,
                    "stem_routing": [
                        {"name": item["name"], "role": item["role"]}
                        for item in groups
                    ],
                },
                "resulting_characteristics": report[
                    "naturalness_characteristics"
                ],
            },
        }
    )
    if report["status"] != "completed":
        return report

    safe_song = slug(song)
    naturalized_path = job_dir / f"{safe_song}_Naturalized.wav"
    original_path = job_dir / f"{safe_song}_Original_AB_Reference.wav"
    pre_denoise_path = job_dir / f"{safe_song}_PreDenoise_AB_Reference.wav"
    report_path = job_dir / f"{safe_song}_Naturalize_Report.json"

    write_audio(naturalized_path, processed, sample_rate, allow_clip=False)
    if bool(payload.get("retain_original", True)):
        write_audio(original_path, source, sample_rate, allow_clip=False)
    if bool(payload.get("retain_pre_denoise", True)):
        safe_pre, _ = apply_headroom(pre_denoise)
        write_audio(pre_denoise_path, safe_pre, sample_rate, allow_clip=False)

    report["processed_path"] = str(naturalized_path)
    report["original_reference_path"] = (
        str(original_path) if original_path.exists() else None
    )
    report["pre_denoise_reference_path"] = (
        str(pre_denoise_path) if pre_denoise_path.exists() else None
    )
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if bool(payload.get("publish_outputs", True)):
        outputs = [naturalized_path, report_path]
        if original_path.exists():
            outputs.insert(1, original_path)
        if pre_denoise_path.exists():
            outputs.insert(-1, pre_denoise_path)
        report["outputs"] = publish_files(
            outputs,
            artist=artist,
            category="naturalized",
            ttl_seconds=int(payload.get("output_ttl_seconds", 86400)),
        )
    return report
