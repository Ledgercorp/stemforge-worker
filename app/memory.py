from __future__ import annotations

import json
import os
import re
from pathlib import Path


MEMORY_DIR = Path(
    os.environ.get("STEMFORGE_WORKSPACE", "/tmp/stemforge")
) / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "default"


def _path(artist: str) -> Path:
    return MEMORY_DIR / f"{_slug(artist)}.json"


def get_memory(artist: str) -> dict:
    path = _path(artist)
    if not path.exists():
        return {
            "artist": artist,
            "version": 1,
            "preferences": {
                "avoid_vocal_time_warping": True,
                "prefer_stereo_mix_as_coherence_anchor": True,
                "metadata_policy": "clean audio export plus JSON sidecar",
            },
            "feedback": [],
            "approved_lyric_timings": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_memory(artist: str, memory: dict) -> Path:
    path = _path(artist)
    memory["version"] = int(memory.get("version", 0)) + 1
    path.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    return path


def record_feedback(payload: dict) -> dict:
    artist = payload.get("artist", "sounddecay")
    feedback = str(payload.get("feedback") or "").strip()
    if not feedback:
        return {"status": "rejected", "reason": "feedback is required."}

    memory = get_memory(artist)
    memory.setdefault("feedback", []).append(
        {
            "song": payload.get("song"),
            "feedback": feedback,
            "preferred_version": payload.get("preferred_version"),
            "rejected_version": payload.get("rejected_version"),
        }
    )
    path = save_memory(artist, memory)
    return {"status": "completed", "memory_path": str(path), "memory": memory}


def save_approved_timing(artist: str, song: str, timing: dict) -> Path:
    memory = get_memory(artist)
    memory.setdefault("approved_lyric_timings", {})[song] = timing
    return save_memory(artist, memory)
