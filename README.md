# StemForge RunPod Worker

A private, GitHub-ready RunPod Serverless worker for StemForge.

This starter build includes:

- WhisperX lyric transcription and word alignment
- reconciliation against exact supplied lyrics
- line-level confidence and review flags
- SRT, ASS karaoke, LRC, CSV, and JSON export
- audio analysis
- basic objective A/B comparison
- persistent artist memory on a RunPod network volume
- API actions for future mixing and mastering tools

## Repository structure

```text
stemforge-runpod/
├── Dockerfile
├── requirements.txt
├── handler.py
├── .gitignore
├── README.md
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── audio_analysis.py
│   ├── export.py
│   ├── lyric_align.py
│   ├── mastering.py
│   ├── memory.py
│   ├── perceptual_listener.py
│   ├── stem_processing.py
│   └── utils.py
├── models/
├── output/
├── reports/
└── workspace/
```

## Upload to GitHub

1. Create a private GitHub repository.
2. Upload the **contents of this folder**, not the outer ZIP.
3. Confirm `Dockerfile` is in the repository root.
4. Commit the files.

## Deploy on RunPod

1. Create or connect a RunPod account.
2. Connect the private GitHub repository.
3. Create a Serverless endpoint from the repository.
4. Use the root `Dockerfile`.
5. Attach a network volume mounted at `/runpod-volume`.
6. Choose a GPU worker with at least 16 GB VRAM.
7. Set minimum workers to `0` and maximum workers to `1`.
8. Increase execution timeout for full-song alignment.

The first job downloads Whisper and alignment models into the persistent volume. Later jobs reuse them.

## Request format

The default action is `align_lyrics`.

```json
{
  "input": {
    "action": "align_lyrics",
    "audio_url": "https://example.com/final-master.wav",
    "vocal_url": "https://example.com/lead-vocal.wav",
    "lyrics": "First lyric line\nSecond lyric line",
    "artist": "sounddecay",
    "song": "Sweet Sixteen",
    "model": "large-v3",
    "language": "en",
    "approve": false
  }
}
```

When `vocal_url` is supplied, StemForge aligns against it. Otherwise it uses `audio_url`.

## Other actions

```text
analyze_audio
compare_audio
get_memory
record_feedback
mastering_plan
inspect_stems
```

## Important limitation

WhisperX is much better on isolated vocals than on a dense shoegaze master. Heavy distortion, screaming, reverb, layered vocals, or altered pronunciation can reduce recognition accuracy. The engine marks weakly anchored lines for review rather than pretending they are exact.

## Secrets

Do not commit API keys to this repository. Keep RunPod and storage credentials in private environment variables or secret storage.
