#!/usr/bin/env python3
"""Run the standard ingest client with resilient long polling."""

from __future__ import annotations

import http.client
import socket
import ssl
import time
import urllib.error

import ingest_audio

# Cold or congested serverless workers can take a while even for the small
# create_upload/create_download signing actions.
ingest_audio.POLL_ATTEMPTS = 600

# GitHub-hosted runners occasionally lose the TLS connection while polling
# api.runpod.ai. The RunPod job itself is still alive, so abandoning the poll
# would strand a valid upload authorization and cause duplicate jobs on retry.
_real_urlopen = ingest_audio.urllib.request.urlopen
_TRANSIENT = (
    urllib.error.URLError,
    ssl.SSLError,
    socket.timeout,
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    http.client.RemoteDisconnected,
)


def _resilient_urlopen(*args, **kwargs):
    last_error: BaseException | None = None
    for attempt in range(30):
        try:
            return _real_urlopen(*args, **kwargs)
        except _TRANSIENT as exc:
            last_error = exc
            if attempt == 29:
                raise
            # Cap the delay so one unstable status endpoint gets several
            # minutes to recover without making healthy calls unnecessarily slow.
            time.sleep(min(3 + attempt * 2, 20))
    assert last_error is not None
    raise last_error  # pragma: no cover


ingest_audio.urllib.request.urlopen = _resilient_urlopen

if __name__ == "__main__":
    raise SystemExit(ingest_audio.main())
