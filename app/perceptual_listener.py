from __future__ import annotations

from app.audio_analysis import analyze_audio_url


def compare_audio_urls(payload: dict) -> dict:
    source_url = payload.get("source_url")
    candidate_url = payload.get("candidate_url")

    if not source_url or not candidate_url:
        return {
            "status": "rejected",
            "reason": "source_url and candidate_url are required.",
        }

    source = analyze_audio_url({"audio_url": source_url})
    candidate = analyze_audio_url({"audio_url": candidate_url})

    if source.get("status") != "completed" or candidate.get("status") != "completed":
        return {
            "status": "error",
            "source": source,
            "candidate": candidate,
        }

    source_metrics = source["analysis"]
    candidate_metrics = candidate["analysis"]

    differences = {
        key: round(candidate_metrics[key] - source_metrics[key], 4)
        for key in ["peak_dbfs", "rms_dbfs", "crest_factor_db"]
    }

    return {
        "status": "completed",
        "listener_type": "DSP-based perceptual proxy",
        "honesty_note": (
            "This is not a human-equivalent listener. "
            "It compares measurable characteristics and must not be represented as subjective hearing."
        ),
        "source": source,
        "candidate": candidate,
        "differences": differences,
    }
