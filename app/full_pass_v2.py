from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path

from app.github_delivery import _upload_release_asset
from app.mastering_v2 import master_audio_job
from app.naturalize_v2 import naturalize_audio_job
from app.stems_v2 import inspect_stems_v2_job
from app.storage import WORKSPACE, materialize_input, publish_file


MAX_MIDI_FILES = 64
MAX_MIDI_UNCOMPRESSED_BYTES = 250_000_000


def _stem_specs(payload: dict) -> list[dict]:
    stems = payload.get("stems") or payload.get("stem_urls") or []
    if isinstance(stems, dict):
        stems = [{"name": name, "url": url} for name, url in stems.items()]

    specs: list[dict] = []
    for index, item in enumerate(stems):
        if isinstance(item, str):
            specs.append({"name": f"stem_{index + 1}", "url": item})
        elif isinstance(item, dict):
            specs.append(
                {
                    "name": str(item.get("name") or f"stem_{index + 1}"),
                    **item,
                }
            )
    return specs


def _materialize_stems(specs: list[dict], job_root: Path) -> list[tuple[str, Path]]:
    outputs: list[tuple[str, Path]] = []
    stems_dir = job_root / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    for index, spec in enumerate(specs):
        proxy: dict = {}
        for suffix in ("url", "storage_key", "base64", "path"):
            value = spec.get(suffix)
            if value is not None:
                proxy[f"stem_{suffix}"] = value

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            str(spec.get("name") or f"stem_{index + 1}"),
        ).strip("._-") or f"stem_{index + 1}"

        destination = stems_dir / f"{index:02d}_{safe_name}.wav"
        path = materialize_input(proxy, "stem", destination)
        outputs.append((str(spec.get("name") or safe_name), path))
    return outputs


def _safe_extract_midi(archive_path: Path, destination: Path) -> list[Path]:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total = 0

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            item
            for item in archive.infolist()
            if not item.is_dir()
            and Path(item.filename).suffix.lower() in {".mid", ".midi"}
        ]
        if len(members) > MAX_MIDI_FILES:
            raise ValueError(
                f"MIDI archive contains {len(members)} files; maximum is {MAX_MIDI_FILES}."
            )

        for item in members:
            total += int(item.file_size)
            if total > MAX_MIDI_UNCOMPRESSED_BYTES:
                raise ValueError("MIDI archive exceeds the maximum uncompressed size.")

            target = (destination / Path(item.filename).name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe MIDI archive path: {item.filename}")

            with archive.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            extracted.append(target)

    return sorted(extracted)


def _analyze_midi(paths: list[Path]) -> dict:
    if not paths:
        return {
            "file_count": 0,
            "files": [],
            "primary_tempo_bpm": None,
            "time_signatures": [],
        }

    try:
        import mido
    except ImportError as exc:
        raise RuntimeError("mido is required for MIDI analysis.") from exc

    reports: list[dict] = []
    global_tempos: list[float] = []
    global_signatures: set[str] = set()

    for path in paths:
        midi = mido.MidiFile(str(path))
        tempos: list[float] = []
        signatures: set[str] = set()
        note_on_count = 0
        program_changes: list[int] = []
        track_names: list[str] = []

        for track in midi.tracks:
            track_name = str(getattr(track, "name", "") or "").strip()
            if track_name:
                track_names.append(track_name)

            for message in track:
                if message.type == "note_on" and int(message.velocity) > 0:
                    note_on_count += 1
                elif message.type == "set_tempo":
                    bpm = round(float(mido.tempo2bpm(message.tempo)), 4)
                    tempos.append(bpm)
                    global_tempos.append(bpm)
                elif message.type == "time_signature":
                    signature = (
                        f"{int(message.numerator)}/{int(message.denominator)}"
                    )
                    signatures.add(signature)
                    global_signatures.add(signature)
                elif message.type == "program_change":
                    program_changes.append(int(message.program))

        try:
            duration_seconds = round(float(midi.length), 3)
        except Exception:
            duration_seconds = None

        reports.append(
            {
                "filename": path.name,
                "format_type": int(midi.type),
                "ticks_per_beat": int(midi.ticks_per_beat),
                "track_count": len(midi.tracks),
                "duration_seconds": duration_seconds,
                "note_on_count": note_on_count,
                "track_names": track_names,
                "tempo_bpm_values": sorted(set(tempos)),
                "time_signatures": sorted(signatures),
                "program_changes": sorted(set(program_changes)),
            }
        )

    primary_tempo = None
    if global_tempos:
        rounded = [round(value, 3) for value in global_tempos]
        primary_tempo = max(set(rounded), key=rounded.count)

    return {
        "file_count": len(reports),
        "files": reports,
        "primary_tempo_bpm": primary_tempo,
        "time_signatures": sorted(global_signatures),
    }


def _tempo_profile_overrides(midi_report: dict) -> dict:
    bpm = midi_report.get("primary_tempo_bpm")
    if not bpm or float(bpm) <= 0:
        return {}

    beat_ms = 60000.0 / float(bpm)

    def clamp(value: float, minimum: float, maximum: float) -> float:
        return round(max(minimum, min(value, maximum)), 3)

    return {
        "dynamic": {
            "release_ms": clamp(beat_ms * 0.40, 180.0, 320.0),
        },
        "balanced": {
            "release_ms": clamp(beat_ms * 0.32, 150.0, 260.0),
        },
        "dense": {
            "release_ms": clamp(beat_ms * 0.25, 120.0, 220.0),
        },
    }


def _compact_stem_report(report: dict) -> dict:
    return {
        "stem_count": report.get("stem_count"),
        "possible_duplicates": report.get("possible_duplicates"),
        "reconstruction": report.get("reconstruction"),
        "stems": [
            {
                "name": item.get("name"),
                "metrics": {
                    key: (item.get("metrics") or {}).get(key)
                    for key in (
                        "duration_seconds",
                        "sample_rate",
                        "channels",
                        "integrated_lufs",
                        "true_peak_dbtp",
                        "sample_peak_dbfs",
                        "rms_dbfs",
                        "crest_factor_db",
                        "stereo_correlation",
                        "clipped_sample_count",
                    )
                },
                "estimated_timeline_offset": item.get(
                    "estimated_timeline_offset"
                ),
                "silent": item.get("silent"),
                "clipped": item.get("clipped"),
            }
            for item in report.get("stems") or []
        ],
    }


def _github_config(payload: dict) -> tuple[str, int, str] | None:
    config = payload.get("github_release_export")
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError("github_release_export must be an object.")

    repository = str(config.get("repository") or "").strip()
    release_id = int(config.get("release_id") or 0)
    token = str(config.get("token") or "").strip()

    if "/" not in repository or release_id <= 0 or not token:
        raise ValueError("Invalid github_release_export configuration.")
    return repository, release_id, token


def _naturalize_config(value: object) -> dict | None:
    if value in (None, False):
        return None
    if value is True:
        return {}
    if isinstance(value, str):
        return {"mode": value}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("naturalize must be true, false, a mode string, or an object.")


def _compact_naturalize_report(report: dict) -> dict:
    return {
        "applied": report.get("status") == "completed",
        "engine": report.get("engine"),
        "label": report.get("label"),
        "requested_mode": report.get("requested_mode"),
        "resolved_mode": report.get("resolved_mode"),
        "intensity": report.get("intensity"),
        "processing_order": report.get("processing_order"),
        "nominal_cocktail": report.get("nominal_cocktail"),
        "naturalness_characteristics": report.get(
            "naturalness_characteristics"
        ),
        "reconstruction_gate": report.get("reconstruction_gate"),
        "anchor_decision": report.get("anchor_decision"),
        "safety": report.get("safety"),
        "agent_log": report.get("agent_log"),
        "honesty_note": report.get("honesty_note"),
    }


def full_pass_job(payload: dict) -> dict:
    artist = str(payload.get("artist") or "sounddecay")
    song = str(payload.get("song") or "untitled")
    profiles = payload.get("profiles") or ["dynamic", "balanced", "dense"]
    if isinstance(profiles, str):
        profiles = [profiles]

    specs = _stem_specs(payload)
    if not specs:
        return {
            "status": "rejected",
            "reason": "stems must contain at least one stem.",
        }

    try:
        naturalize_config = _naturalize_config(payload.get("naturalize"))
    except ValueError as exc:
        return {"status": "rejected", "reason": str(exc)}

    job_id = uuid.uuid4().hex[:16]
    job_root = (WORKSPACE / "jobs" / "full_pass" / job_id).resolve()
    job_root.mkdir(parents=True, exist_ok=True)

    master_path = materialize_input(
        payload,
        "master",
        job_root / "master_source.wav",
    )
    stems = _materialize_stems(specs, job_root)

    midi_paths: list[Path] = []
    midi_archive = materialize_input(
        payload,
        "midi_zip",
        job_root / "midi_input.zip",
        required=False,
    )
    if midi_archive is not None:
        midi_paths = _safe_extract_midi(
            midi_archive,
            job_root / "midi",
        )

    midi_report = _analyze_midi(midi_paths)
    tempo_overrides = _tempo_profile_overrides(midi_report)

    stem_report = inspect_stems_v2_job(
        {
            "artist": artist,
            "song": song,
            "master_path": str(master_path),
            "stems": [
                {"name": name, "path": str(path)}
                for name, path in stems
            ],
            "output_ttl_seconds": int(
                payload.get("output_ttl_seconds", 86400)
            ),
        }
    )
    if stem_report.get("status") != "completed":
        return stem_report

    reconstruction = stem_report.get("reconstruction") or {}
    safe_for_remix = bool(reconstruction.get("safe_for_stem_remix"))
    anchor_decision = {
        "selected_anchor": "stereo_master",
        "stem_sum_safe_for_remix": safe_for_remix,
        "reason": (
            "The stems passed the reconstruction threshold, but this conservative "
            "pass still retains the supplied stereo master as the mix anchor."
            if safe_for_remix
            else "The stems did not pass the reconstruction threshold, so the "
            "supplied stereo master was retained as the coherence anchor."
        ),
    }

    naturalize_report = None
    mastering_input_path = master_path
    if naturalize_config is not None:
        naturalize_payload = {
            "action": "naturalize",
            "artist": artist,
            "song": song,
            "audio_path": str(master_path),
            "mode": naturalize_config.get("mode", "auto"),
            "intensity": naturalize_config.get(
                "intensity",
                naturalize_config.get("depth", 0.35),
            ),
            "retain_original": naturalize_config.get("retain_original", True),
            "publish_outputs": False,
            "separate_if_needed": naturalize_config.get(
                "separate_if_needed",
                False,
            ),
            "separation_model": naturalize_config.get(
                "separation_model",
                "htdemucs",
            ),
            "job_dir": str(job_root / "naturalize"),
            "stems": [
                {"name": name, "path": str(path)}
                for name, path in stems
            ],
        }
        naturalize_report = naturalize_audio_job(naturalize_payload)
        if naturalize_report.get("status") != "completed":
            return {
                "status": "rejected",
                "reason": "The requested Naturalize stage failed its safety gate.",
                "naturalize": naturalize_report,
            }
        mastering_input_path = Path(
            str(naturalize_report["processed_path"])
        )

    mastering_payload = {
        "artist": artist,
        "song": song,
        "audio_path": str(mastering_input_path),
        "profiles": profiles,
        "profile_overrides": tempo_overrides,
        "output_ttl_seconds": int(
            payload.get("output_ttl_seconds", 86400)
        ),
    }
    if payload.get("github_release_export"):
        mastering_payload["github_release_export"] = payload[
            "github_release_export"
        ]

    mastering = master_audio_job(mastering_payload)
    if mastering.get("status") != "completed":
        return mastering

    compact_candidates = [
        {
            "profile": item.get("profile"),
            "chain": item.get("chain"),
            "metrics": item.get("metrics"),
            "safety": item.get("safety"),
            "output": item.get("output"),
        }
        for item in mastering.get("candidates") or []
    ]

    report = {
        "status": "completed",
        "engine": "StemForge full-pass engine v2.1",
        "artist": artist,
        "song": song,
        "input_summary": {
            "master_filename": master_path.name,
            "stem_count": len(stems),
            "stem_names": [name for name, _ in stems],
            "midi_file_count": len(midi_paths),
            "midi_filenames": [path.name for path in midi_paths],
        },
        "midi_analysis": midi_report,
        "tempo_linked_mastering_overrides": tempo_overrides,
        "stem_integrity": _compact_stem_report(stem_report),
        "anchor_decision": anchor_decision,
        "naturalize": (
            _compact_naturalize_report(naturalize_report)
            if naturalize_report is not None
            else {"applied": False}
        ),
        "mastering_source": {
            "path": str(mastering_input_path),
            "naturalized_before_mastering": naturalize_report is not None,
        },
        "source_metrics": mastering.get("source_metrics"),
        "candidates": compact_candidates,
        "mastering_candidates": compact_candidates,
        "mastering_report_output": mastering.get("report_output"),
        "honesty_note": (
            "The stems and MIDI informed integrity checks, tempo-linked compressor "
            "release timing, and the coherence-anchor decision. When Naturalize "
            "was requested, its non-destructive safety-approved render became the "
            "mastering source. Naturalize is an authenticity enhancement, not an "
            "audio-origin detection or concealment tool."
        ),
    }

    safe_song = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        song,
    ).strip("._-") or "untitled"
    report_path = job_root / f"{safe_song}_FullPass_Report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    github = _github_config(payload)
    if github is not None:
        repository, release_id, token = github
        asset = _upload_release_asset(
            repository=repository,
            release_id=release_id,
            token=token,
            path=report_path,
            asset_name=f"{safe_song}_FullPass_Report.json",
        )
        report["full_pass_report_output"] = {
            **asset,
            "storage_mode": "private_github_release_asset",
        }
    else:
        report["full_pass_report_output"] = publish_file(
            report_path,
            artist=artist,
            category="full_pass_reports",
            ttl_seconds=int(payload.get("output_ttl_seconds", 86400)),
        )

    return report
