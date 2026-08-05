#!/usr/bin/env python3
"""Run the standard ingest client with resilient long polling."""

from __future__ import annotations

import ssl
import time
import urllib.error

import ingest_audio

# The endpoint can take more than five minutes to return a lightweight
# create_upload/create_download result during a cold or congested period.
ingest_audio.POLL_ATTEMPTS = 300

# GitHub-hosted runners occasionally see a transient TLS EOF while polling
# api.runpod.ai. A single dropped status request must not abandon a valid
# worker job or create a duplicate upload authorization.
_real_urlopen = ingest_audio.urllib.request.urlopen


def _resilient_urlopen(*args, **kwargs):
    last_error = None
    for attempt in range(6):
        try:
            return _real_urlopen(*args, **kwargs)
        except (urllib.error.URLError, ssl.SSLError) as exc:
            last_error = exc
            if attempt == 5:
                raise
            time.sleep(2 + attempt * 2)
    raise last_error  # pragma: no cover


ingest_audio.urllib.request.urlopen = _resilient_urlopen

if __name__ == "__main__":
    raise SystemExit(ingest_audio.main())
