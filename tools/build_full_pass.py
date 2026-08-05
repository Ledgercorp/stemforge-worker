#!/usr/bin/env python3
"""Turn an ingest record into a full_pass request.

`tools/ingest_audio.py` stores a song's files and writes their storage keys to
`ingested/<song>.json`. This reads that record and emits the job that renders
from those keys, so the path from "here are my files" to "here is the render"
has no hand-copied key in it - which is where the Drive-link era lost its
inputs.

    python tools/build_full_pass.py ingested/sweet-sixteen.json \\
        --song "Sweet Sixteen" --out full_pass_direct_requests/sweet-sixteen.json

Each stored file is classified by filename: the master, the MIDI archive, and
everything else as a stem. Stems are reported with the role they will route to
under Naturalize, because a stem whose name says nothing recognisable falls to
`other` and is processed differently - better to see that here than in a
finished render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# The settings the delivered Hypervigilant pass ran with. Naturalize is a
# fidelity enhancement; the worker's quality gates decide what survives, and
# nothing here weakens them.
NATURALIZE = {
    "mode": "auto",
    "intensity": 1.4,
    "passes": 1,
    "vocoder": "auto",
    "parameters": {"denoise_reduction_db": 3.0, "noise_profile_ms": 200},
    "retain_original": True,
    "retain_pre_denoise": True,
}
PROFILES = ["dynamic", "balanced", "dense"]
REQUIRED_BUILD = "v2.4.0-naturalize-balanced-vocoder"
MINIMUM_VERSION = "2.4.0"
POLICY = {"executionTimeout": 5_400_000, "ttl": 7_200_000}

AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".mp3", ".m4a", ".ogg", ".opus"}


class BuildError(RuntimeError):
    """Raised when a record cannot produce a runnable job."""


def _stem_name(filename: str) -> str:
    """`Lead_Vocals.wav` -> `Lead Vocals`, leaving anything else readable."""
    return Path(filename).stem.replace("_", " ").strip() or Path(filename).stem


def _role_of(name: str) -> str:
    """The role the worker will assign, or "" when it cannot be checked here.

    Imported rather than restated: a second copy of the vocabulary would drift
    from the one that actually runs, and silently mis-report roles.
    """
    try:
        from app.naturalize_dsp_v2 import role_from_name
    except ImportError:  # numpy absent - the job is still correct without this
        return ""
    return role_from_name(name)


def classify(records: list[dict]) -> tuple[dict, dict | None, list[dict]]:
    """Split stored files into the master, the MIDI archive, and the stems."""
    master: dict | None = None
    midi: dict | None = None
    stems: list[dict] = []

    for record in records:
        filename = record.get("filename") or ""
        lowered = filename.lower()
        suffix = Path(lowered).suffix

        if suffix == ".zip":
            if midi is not None:
                raise BuildError(
                    f"two archives in the record ({midi['filename']} and "
                    f"{filename}); only one can be the MIDI archive."
                )
            midi = record
        elif "master" in lowered:
            if master is not None:
                raise BuildError(
                    f"two files look like the master ({master['filename']} and "
                    f"{filename}); rename one so only one matches."
                )
            master = record
        elif suffix in AUDIO_SUFFIXES:
            stems.append(record)
        else:
            raise BuildError(f"unrecognised file in the record: {filename}")

    if master is None:
        raise BuildError(
            "no master found. One stored filename must contain 'master' - "
            "pass it through the ingest's `names` input if the source URL "
            "does not carry a filename."
        )
    if not stems:
        raise BuildError("no stems found in the record.")
    return master, midi, stems


def build(
    records: list[dict],
    *,
    song: str,
    artist: str = "sounddecay",
    output_ttl_seconds: int = 86_400,
) -> tuple[dict, list[tuple[str, str]]]:
    """Return the job payload and the (stem name, role) routing it implies."""
    master, midi, stems = classify(records)

    stem_specs = []
    routing = []
    for record in stems:
        name = _stem_name(record["filename"])
        stem_specs.append({"name": name, "storage_key": record["storage_key"]})
        routing.append((name, _role_of(name)))

    payload = {
        "action": "full_pass",
        "artist": artist,
        "song": song,
        "master_storage_key": master["storage_key"],
        "stems": stem_specs,
        "profiles": list(PROFILES),
        "naturalize": dict(NATURALIZE),
        "output_ttl_seconds": output_ttl_seconds,
        "required_build": REQUIRED_BUILD,
        "minimum_worker_version": MINIMUM_VERSION,
    }
    if midi is not None:
        payload["midi_zip_storage_key"] = midi["storage_key"]

    return {"input": payload, "policy": dict(POLICY)}, routing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a full_pass request from an ingest record."
    )
    parser.add_argument("record", help="Path to ingested/<song>.json")
    parser.add_argument("--song", required=True, help="Song name for the job.")
    parser.add_argument("--artist", default="sounddecay")
    parser.add_argument("--out", help="Where to write the job (default: stdout).")
    parser.add_argument("--output-ttl-seconds", type=int, default=86_400)
    args = parser.parse_args(argv)

    try:
        record_path = Path(args.record)
        loaded = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read {args.record}: {exc}", file=sys.stderr)
        return 1

    records = loaded.get("ingested") if isinstance(loaded, dict) else loaded
    if not records:
        print(f"{args.record} lists no ingested files.", file=sys.stderr)
        return 1

    try:
        job, routing = build(
            records,
            song=args.song,
            artist=args.artist,
            output_ttl_seconds=args.output_ttl_seconds,
        )
    except BuildError as exc:
        print(f"{args.record}: {exc}", file=sys.stderr)
        return 1

    for name, role in routing:
        if role:
            flag = "  <- no role keyword; treated as 'other'" if role == "other" else ""
            print(f"  {name:<24} {role}{flag}", file=sys.stderr)

    text = json.dumps(job, indent=2) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(args.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
