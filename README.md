# StemForge v2

StemForge is a private RunPod Serverless audio-production worker operated through the GitHub relay.

## Live capabilities

### Lyrics

- WhisperX `large-v3` transcription and word alignment
- monotonic reconciliation against exact supplied lyrics
- section-label removal and repeated-chorus preservation
- intro-skip and first-visible-line rules
- final-master duration clamping and gap-aware caption endings
- SRT, word-timed ASS, LRC, CSV and JSON exports

### Audio analysis and comparison

- full-song integrated loudness, true peak, RMS and crest factor
- clipping, discontinuity, DC offset and silence-region detection
- stereo correlation, transient density and spectral distribution
- ten-second metric timeline
- waveform and spectrogram rendering
- level-matched objective A/B residual and safety comparison

Objective comparison is a DSP proxy, not human hearing.

### Stems and separation

- per-stem technical inspection
- timeline-offset estimates
- duplicate-stem fingerprinting
- master reconstruction and null-residual testing
- remix-safety flagging
- Demucs source separation, clearly labeled as estimated stems

### Processing

- dynamic, balanced and dense 24-bit mastering candidates
- post-render safety rejection for clipping, phase risk and excessive crest loss
- conservative click repair, hum notching and restrained harshness control
- deterministic phrase microdynamics and transient variation
- no vocal time warping or pitch correction

### Transfer and delivery

- URL, base64, S3 storage key and restricted RunPod-volume inputs
- optional private expiring S3-compatible upload and download links
- automatic output publishing when storage credentials are configured

Signed transfer requires:

```text
STEMFORGE_S3_BUCKET
STEMFORGE_S3_ENDPOINT_URL
STEMFORGE_S3_REGION
STEMFORGE_S3_ACCESS_KEY_ID
STEMFORGE_S3_SECRET_ACCESS_KEY
```

Without those variables, files remain on the attached RunPod volume.

### Video and DAW

- deterministic horizontal, vertical and square lyric-video rendering through FFmpeg
- supplied real background media, subtitles, logo, grain and slow motion
- no generative narrative footage
- Reaper RPP, marker CSV, MIDI marker track, chapter text and ZIP export

### Persistent memory

- raw feedback and approved lyric timings
- structured production rules with global or song scope
- confidence, provenance and confirmation counts
- applicable production-profile retrieval for later jobs

## Main actions

```text
system_info
storage_info
create_upload
create_download
delete_storage_objects
align_lyrics
align_lyrics_smart
analyze_audio_v2
compare_audio_v2
inspect_stems_v2
separate_stems
mastering_plan
master_audio
repair_audio
humanize_audio
render_lyric_video
export_daw
get_memory
record_feedback
record_rule
get_production_profile
```

Legacy v1 analysis and alignment actions remain available for compatibility.

## Boundaries

- Demucs outputs are not original multitracks.
- Mastering candidates require level-matched listening before approval.
- Offset estimates may be weak on ambient or sparse stems.
- Direct live control of Logic is not included. DAW integration is currently file-based.
- Removing metadata does not prove or conceal how audio was created.
