# StemForge v2

StemForge v2 turns the original lyric-alignment worker into a broader, deterministic audio-production service.

## Working actions

### Transfer and output delivery

- `create_upload`: creates a private expiring S3-compatible PUT URL.
- `create_download`: creates a private expiring GET URL.
- `delete_storage_objects`: removes temporary inputs and outputs.
- Every render action returns signed download URLs when storage is configured, otherwise it retains RunPod-volume paths.

Required optional environment variables:

```text
STEMFORGE_S3_BUCKET
STEMFORGE_S3_ENDPOINT_URL
STEMFORGE_S3_REGION
STEMFORGE_S3_ACCESS_KEY_ID
STEMFORGE_S3_SECRET_ACCESS_KEY
```

### Lyrics

- `align_lyrics`: existing WhisperX exact-lyric reconciliation.
- `align_lyrics_smart`: removes section labels, preserves repeated choruses and all-caps breakdowns, supports `skip_intro_lines`, `first_display_text`, final-master duration clamping, maximum caption duration, and gap-aware caption endings.

### Analysis

- `analyze_audio_v2`: full-song LUFS, true peak, RMS, crest factor, clipping, discontinuities, stereo correlation, transient density, spectral distribution, silence regions, ten-second timeline, waveform and spectrogram.
- `compare_audio_v2`: level-matched residual similarity plus clipping, true-peak, phase and crest-loss safety checks.

### Stems

- `inspect_stems_v2`: downloads and normalizes stems, measures each one, estimates timeline offsets, identifies likely duplicates, reconstructs the mix when a master is provided, and renders the null residual.
- `separate_stems`: Demucs separation with normal or two-stem modes. Separated files are explicitly labeled as estimates.

### Processing

- `master_audio`: renders dynamic, balanced and dense 24-bit candidates, measures every result, and rejects objectively unsafe candidates.
- `repair_audio`: conservative de-clicking, optional 50/60 Hz hum notching, narrow-ridge diagnostics and restrained harshness control.
- `humanize_audio`: deterministic length-preserving microdynamics and transient variation. Lead-vocal processing is blocked by default and no time warping is used.

### Video and DAW

- `render_lyric_video`: deterministic FFmpeg rendering in vertical, horizontal and square formats using supplied real backgrounds, audio, subtitles and optional logo.
- `export_daw`: Reaper RPP, marker CSV, MIDI marker track, chapters and ZIP package.

### Memory

- `record_rule`: records structured rules with scope, song, provenance, confidence and confirmation count.
- `get_production_profile`: returns the applicable global and song-specific production rules.

## Important boundaries

- Objective comparison is not human hearing.
- Demucs stems are not original multitracks.
- The mastering candidates are real renders, but final selection still requires level-matched listening.
- Secure signed transfer is coded but requires S3-compatible storage credentials.
- Direct live control of Logic is not included. DAW interchange is file-based; a future Reaper render node can apply plugin chains headlessly.
