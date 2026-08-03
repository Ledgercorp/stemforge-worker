from __future__ import annotations


def inspect_stems_job(payload: dict) -> dict:
    stem_urls = payload.get("stem_urls") or []
    if not stem_urls:
        return {
            "status": "rejected",
            "reason": "stem_urls must contain at least one URL.",
        }

    return {
        "status": "completed",
        "stem_count": len(stem_urls),
        "classification": "analysis_pending",
        "note": (
            "This starter repository accepts stem URLs but does not yet download and "
            "perform reconstruction/null testing. Add that before allowing automatic stem-based remixing."
        ),
    }
