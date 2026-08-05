#!/usr/bin/env python3
"""Run the standard ingest client with a longer RunPod cold-start window."""

from __future__ import annotations

import ingest_audio

# The endpoint can take more than five minutes to return a lightweight
# create_upload/create_download result during a cold or congested period.
# Keep all ingest behavior identical and extend only the polling window.
ingest_audio.POLL_ATTEMPTS = 300

if __name__ == "__main__":
    raise SystemExit(ingest_audio.main())
