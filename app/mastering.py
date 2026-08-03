from __future__ import annotations

from app.memory import get_memory


def mastering_plan(payload: dict) -> dict:
    artist = payload.get("artist", "sounddecay")
    memory = get_memory(artist)

    return {
        "status": "completed",
        "mode": "planning_only",
        "artist": artist,
        "memory_used": memory,
        "plan": [
            "Validate source format and duration.",
            "Evaluate whether supplied stems reconstruct the stereo master safely.",
            "Retain the stereo mix as the coherence anchor when reconstruction is imperfect.",
            "Generate conservative, balanced, and dynamic preview candidates.",
            "Compare candidates against the source using objective safety checks.",
            "Render the final master only after candidate approval.",
        ],
        "note": (
            "This starter repository includes the planning interface. "
            "Full mastering render chains should be added as explicit, reversible DSP operations."
        ),
    }
