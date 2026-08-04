# StemForge v2.4

StemForge is a private RunPod Serverless audio-production worker operated through a guarded GitHub relay.

## Release identity

```text
stemforge_version: 2.4.0
build: v2.4.0-naturalize-balanced-vocoder
```

A GitHub release or merge does not by itself prove that the live RunPod endpoint has rolled forward. Production work must verify the exact version and build through `system_info` before accepting a render.

## Live capabilities

### Lyrics

- WhisperX `large-v3` transcription and word alignment
- monotonic reconciliation against supplied lyrics
- repeated-section preservation and section-label removal
- intro-skip and first-visible-line controls
- SRT, ASS, LRC, CSV and JSON exports

### Audio analysis and comparison

- integrated loudness, true peak, RMS and crest factor
- clipping, discontinuity, DC-offset and silence detection
- stereo correlation, transient density and spectral distribution
- level-matched A/B residual comparison
- derivative, click, discontinuity and high-frequency artifact gates
- noise-floor-aware high-band analysis with raw diagnostics retained

Objective comparison is a DSP proxy, not a substitute for listening.

### Stems and separation

- per-stem technical inspection
- timeline-offset estimates
- duplicate-stem fingerprinting
- master reconstruction and null-residual testing
- strict remix-safety decisions
- Demucs separation, clearly labeled as estimated stems

### Coherence-safe stem remix

StemForge uses two explicit remix modes:

- `full_stem_mix`: only when supplied stems pass strict reconstruction thresholds
- `master_anchored_delta`: preserves the coherent stereo master while applying controlled stem-derived changes around it
- `auto`: selects full stem replacement only when reconstruction is safe

Auxiliary stems such as bass pads can use `"role": "bass_pad"` or `"overlay": true`. They are excluded from reconstruction fitting and mixed directly at the requested level.

The remix path does not use autonomous stereo widening. Every render is rejected for non-finite samples, clipping, excessive derivative energy, new discontinuities, meaningful high-frequency noise growth or unacceptable reference-correlation loss.

## Naturalize Balanced

`naturalize` is a non-destructive Naturalness / Authenticity enhancement that restores restrained pitch, amplitude, timing, phase and harmonic micro-variation while prioritizing transparency.

Default preset:

```text
Naturalize Balanced
```

### Core cocktail

The order is mandatory:

1. Primary vibrato: 14–16 Hz, default 15 Hz; 8–12 cents, default 10; sine
2. Tremolo: 12–16 Hz, default 15 Hz; 6–12%, default 10%; sine or soft triangle
3. Secondary vibrato: 8–11 Hz, default 10 Hz; 6–10 cents, default 9
4. Flanger: 0.8–1.2 ms, default 1.0 ms; modulation rate 0.6–1.1 Hz, default 0.9 Hz; depth 7–10, default 9; feedback capped at 15%; wet mix 20–40%
5. Optional humanization layers
6. Constrained adaptive denoise after all modulation
7. Optional neural vocoder after denoise

Modulation depth is frequency-dependent: the musical midrange receives more movement while spectral extremes remain lighter.

### Constrained denoise

- adaptive music-oriented spectral denoise only
- always after the cocktail and optional humanization layers
- 3–8 dB maximum reduction, default 5 dB
- noise profile selected from the quietest 200–500 ms after modulation, default 350 ms
- fast attack, slower release and explicit transient protection
- dry preservation component to retain introduced micro-variation
- never used to restore artificial spectral uniformity

### Optional humanization layers

- shaped pink or tape-hiss noise floor between -72 and -60 dBFS
- onset-aware, velocity-sensitive micro-timing jitter between ±1.5 and ±3 ms
- low-drive, gain-compensated tape/tube-style saturation
- optional gentle multiband compression
- frequency-dependent modulation depth

### Stem-aware routing

- vocals: 1.20× default depth, constrained to the 1.1–1.3× policy range
- harmonic instruments: 1.00× default, constrained to 0.9–1.1×
- percussion and drums: 0.65× default, constrained to 0.5–0.8×
- unknown roles: conservative fallback treatment

`auto` always prefers Surgical mode when a vocal stem exists. Quick mode is used only when stem-aware processing is unavailable or explicitly requested.

### Modes

- `quick`: processes the supplied full mix
- `surgical`: processes vocal, harmonic, percussion and other stems independently, then recombines them
- `auto`: prefers Surgical when vocal stems exist and falls back to Quick only when necessary

Surgical mode validates stem reconstruction. Unsafe stems use a master-anchored processing delta instead of replacing the coherent source. If the anchored result still fails the transparency gate, the operation aborts cleanly.

### Intensity and iteration

- global intensity range: 0.6–1.4
- default intensity: 1.0
- explicit `0` is a true bypass
- rates remain fixed while intensity scales depths, wetness and optional feature perturbation
- maximum two passes
- second pass capped at 0.70× the selected intensity
- unsafe results trigger automatic intensity reduction attempts before rejection

### Neural vocoders

Neural reconstruction always occurs after the cocktail and constrained denoise.

- Vocos: default Surgical fast path when installed; default model `charactr/vocos-mel-24khz`
- BigVGAN-v2: explicit higher-quality path; default model `nvidia/bigvgan_v2_44khz_128band_512x`
- DisCoder: explicit environment-provisioned maximum-music-fidelity path

Default wet routing:

- vocals: 0.85
- harmonic instruments: 0.60
- percussion: 0.20
- other stems: 0.45

The intensity scaler controls vocoder wetness and optional feature perturbation. Every vocoder result is independently checked for aliasing, metallic artifacts, transient loss and clarity degradation. Unsafe or unavailable backends fall back to the pure DSP render.

### Safety and A/B behavior

Naturalize retains by default:

- original PCM reference
- pre-denoise intermediate
- final Naturalized render
- complete JSON report

Quality gates detect:

- audible-warble risk
- pumping and envelope damage
- metallic or derivative artifacts
- transient loss
- aliasing or high-frequency growth
- clipping and non-finite samples
- spectral and waveform clarity loss

Reports log every resolved parameter, denoise reduction, quiet-profile selection, noise-floor level, pass intensity, stem role and multiplier, reconstruction decision, vocoder model, wet ratio, feature perturbation, quality attempt and fallback decision.

Naturalize is a fidelity/authenticity enhancement. It is not an audio-origin detector, concealment method or detection-evasion tool.

## Processing and delivery

- dynamic, balanced and dense 24-bit mastering candidates
- balanced and heavy coherence-safe stem-remix candidates
- optional Naturalize stage in autonomous `full_pass` before mastering
- true premaster before loudness normalization and peak trimming
- conservative click repair, hum notching and harshness control
- URL, base64, S3 storage-key and RunPod-volume inputs
- atomic downloads with retries, stale-link detection, optional size and SHA-256 verification
- private GitHub release-asset delivery
- Reaper RPP, marker CSV, MIDI marker track, chapter text and ZIP export

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

## Naturalize request examples

Automatic stem-aware processing:

```json
{
  "input": {
    "action": "naturalize",
    "mode": "auto",
    "intensity": 1.0,
    "passes": 1,
    "audio_url": "https://...",
    "stems": [
      {"name": "Lead Vocals", "role": "vocal", "url": "https://..."},
      {"name": "Guitars", "role": "harmonic", "url": "https://..."},
      {"name": "Drums", "role": "percussion", "url": "https://..."}
    ],
    "vocoder": "auto",
    "retain_original": true,
    "retain_pre_denoise": true
  }
}
```

Advanced Surgical processing:

```json
{
  "input": {
    "action": "naturalize",
    "mode": "surgical",
    "intensity": 1.1,
    "passes": 2,
    "second_pass_intensity": 0.65,
    "audio_url": "https://...",
    "stems": [
      {"name": "Lead Vocals", "role": "vocal", "url": "https://..."},
      {"name": "Instrumental", "role": "harmonic", "url": "https://..."}
    ],
    "parameters": {
      "denoise_reduction_db": 5.0,
      "noise_profile_ms": 350,
      "noise_floor_dbfs": -66,
      "noise_floor_shape": "pink",
      "timing_jitter_ms": 2.25,
      "saturation_drive": 0.10,
      "multiband_compression": false
    },
    "vocoder": {
      "backend": "vocos",
      "feature_perturbation": 0.0025
    }
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
      {"name": "Lead Vocals", "role": "vocal", "url": "https://..."},
      {"name": "Bass", "role": "harmonic", "url": "https://..."},
      {"name": "Drums", "role": "percussion", "url": "https://..."}
    ],
    "naturalize": {
      "mode": "auto",
      "intensity": 1.0,
      "vocoder": "auto",
      "retain_original": true,
      "retain_pre_denoise": true
    }
  }
}
```

Only a safety-approved Naturalized render is passed into mastering.

## Production boundaries

- Demucs outputs are estimates, not original multitracks.
- Unsafe stem replacement is refused; master anchoring does not guarantee acceptance.
- Vocoder availability depends on the deployed image and compute budget.
- A vocoder fallback is not treated as an error when the DSP path passes safely.
- Mastering, remix and Naturalize candidates still require level-matched listening before artistic approval.
- Direct live control of Logic is not included; DAW integration is file-based.
- Metadata removal does not prove or conceal how audio was created.
