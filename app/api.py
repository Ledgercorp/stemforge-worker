from __future__ import annotations

from app.audio_analysis import analyze_audio_url
from app.lyric_align import align_lyrics_job
from app.memory import get_memory, record_feedback
from app.mastering import mastering_plan
from app.perceptual_listener import compare_audio_urls
from app.stem_processing import inspect_stems_job


SUPPORTED_ACTIONS = {
    "align_lyrics",
    "analyze_audio",
    "compare_audio",
    "get_memory",
    "record_feedback",
    "mastering_plan",
    "inspect_stems",
}


def handle_job(payload: dict) -> dict:
    action = payload.get("action", "align_lyrics")

    if action not in SUPPORTED_ACTIONS:
        return {
            "status": "rejected",
            "reason": f"Unsupported action: {action}",
            "supported_actions": sorted(SUPPORTED_ACTIONS),
        }

    if action == "align_lyrics":
        return align_lyrics_job(payload)

    if action == "analyze_audio":
        return analyze_audio_url(payload)

    if action == "compare_audio":
        return compare_audio_urls(payload)

    if action == "get_memory":
        return {
            "status": "completed",
            "memory": get_memory(payload.get("artist", "sounddecay")),
        }

    if action == "record_feedback":
        return record_feedback(payload)

    if action == "mastering_plan":
        return mastering_plan(payload)

    if action == "inspect_stems":
        return inspect_stems_job(payload)

    raise RuntimeError("Unreachable action dispatch.")
