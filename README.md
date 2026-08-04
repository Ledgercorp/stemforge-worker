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
- residual-noise, derivative-growth, click/discontinuity and high-frequency-noise gates
- noise-floor-aware high-band comparison with raw diagnostic values retained

Objective comparison is a DSP proxy, not human hearing.

### Stems and separation

- per-stem technical inspection
- timeline-offset estimates
- duplicate-stem fingerprinting
- master reconstruction and null-residual testing
- strict remix-safety flagging
- Demucs source separation, clearly labeled as estimated stems

### Coherence-safe stem remix

StemForge v2.3.0 supports two explicit remix modes:

- `full_stem_mix`: used only when the supplied stems reconstruct the master to strict null-residual and correlation thresholds
- `master_anchored_delta`: keeps the coherent stereo master as the foundation and applies controlled stem-derived fader and processing changes around it
- `auto`: selects `full_stem_mix` only when the strict reconstruction gate passes; otherwise it selects `master_anchored_delta`

Auxiliary stems such as bass pads can be marked with `"role": "bass_pad"` or `"overlay": true`. They are excluded from reconstruction fitting and mixed directly at the requested gain.

The remix path does not use autonomous stereo widening. Every final render is rejected if it introduces non-finite samples, clipping, excessive derivative energy, new discontinuities, meaningful high-frequency noise growth, or unacceptable reference-correlation loss.

### Naturalize / authenticity processing

`naturalize` is a non-destructive Naturalness / Authenticity enhancement for restoring restrained pitch, amplitude and phase micro-variation.

Mandatory order:

1. 15 Hz vibrato, nominal depth 10
2. 15 Hz tremolo, nominal depth 10
3. 10 Hz vibrato, nominal depth 10
4. 1 ms flanger, modulation 0.90, nominal depth 9
5. light denoise after all modulation

Modes:

- `quick`: applies the complete chain to the full mix
- `surgical`: treats vocals more assertively, applies lighter modulation and level preservation to the instrumental, recombines, then performs final gentle denoise
- `auto`: prefers Surgical when a vocal stem is available and falls back to Quick otherwise

The `intensity` control scales effective depth while retaining the nominal cocktail in the report. Surgical mode validates vocal-plus-instrumental reconstruction and uses a coherent `master - vocal` instrumental anchor when supplied stems are unsafe. Explicit Surgical requests can run Demucs two-stem separation when `separate_if_needed` is enabled.

By default the source is never overwritten and an original PCM A/B reference is retained. Reports include the exact operation order, nominal and effective parameters, mode and anchor decisions, before/after metrics, safety comparisons and measured naturalness characteristics.

Naturalize is a fidelity/authenticity enhancement. It is not an audio-origin detector, concealment method or detection-evasion tool.

### Processing

- dynamic, balanced and dense 24-bit mastering candidates
- balanced and heavy coherence-safe stem-remix candidates
- optional Naturalize stage in autonomous `full_pass` before mastering
- a true premaster generated before final loudness normalization and peak trimming
- post-render safety rejection for clipping, phase risk, artifact growth and excessive crest loss
- conservative click repair, hum notching and restrained harshness control
- deterministic phrase microdynamics and transient variation
- no vocal time warping or pitch correction

### Transfer and delivery

- URL, base64, S3 storage key and restricted RunPod-volume inputs
- atomic URL downloads with retries and stale-link detection
- optional expected byte size and SHA-256 validation for every input
- HTML/browser-confirmation response rejection
- remote URL preflight with ranged-GET fallback when HEAD is unsupported
- optional private expiring S3-compatible upload and download links
- private GitHub release-asset delivery with artifact checksums

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
stem_remix
mastering_plan
master_audio
full_pass
repair_audio
humanize_audio
naturalize
render_lyric_video
export_daw
get_memory
record_feedback
record_rule
get_production_profile
```

Legacy v1 analysis and alignment actions remain available for compatibility.

## Naturalize request guidance

Standalone automatic selection:

```json
{
  "input": {
    "action": "naturalize",
    "mode": "auto",
    "intensity": 0.35,
    "audio_url": "https://...",
    "stems": [
      {"name": "Lead Vocals", "url": "https://..."},
      {"name": "Instrumental", "url": "https://..."}
    ],
    "retain_original": true
  }
}
```

Autonomous full pass:

```json
{
  "input": {
    "action": "full_pass",
    "master_url": "https://...",
    "stems": [
      {"name": "Lead Vocals", "url": "https://..."},
      {"name": "Bass", "url": "https://..."},
      {"name": "Drums", "url": "https://..."}
    ],
    "naturalize": {
      "mode": "auto",
      "intensity": 0.35,
      "retain_original": true
    }
  }
}
```

Only a safety-approved Naturalize render is passed into mastering.

## Stem-remix request guidance

A production request should include:

```json
{
  "input": {
    "action": "stem_remix",
    "remix_mode": "auto",
    "minimum_worker_version": "2.3.0",
    "required_build": "v2.3.0-naturalize-quality",
    "master_url": "https://...",
    "stems": [
      {"name": "Lead Vocals", "role": "lead_vocals", "url": "https://..."},
      {"name": "Breakdown Bass Pad", "role": "bass_pad", "overlay": true, "gain_db": -3.0, "url": "https://..."}
    ],
    "midi_zip_url": "https://...",
    "expected_midi_count": 7,
    "mix_profiles": ["balanced", "heavy"]
  }
}
```

Use `size_bytes` and `sha256` in an input object when the transfer source supports stable integrity metadata.

## Boundaries

- Demucs outputs are estimated stems, not original multitracks.
- A full stem-replacement mix is refused when reconstruction quality is insufficient unless an explicit unsafe override is supplied.
- Mastering, remix and Naturalize candidates still require level-matched listening before artistic approval.
- Offset estimates may be weak on ambient or sparse stems.
- Direct live control of Logic is not included. DAW integration is currently file-based.
- Removing metadata does not prove or conceal how audio was created.
