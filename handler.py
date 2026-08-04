from __future__ import annotations

import runpod

from app.api import handle_job
from app.monotonic_alignment import install as install_monotonic_alignment

install_monotonic_alignment()


def handler(event: dict) -> dict:
    try:
        payload = event.get("input") or {}
        return handle_job(payload)
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
