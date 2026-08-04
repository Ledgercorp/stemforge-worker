# StemForge deployment

The RunPod Serverless endpoint `dyup1dztjr4u15` builds its container image from
this repository's **GitHub source on `main`**. There is no external container
registry, which is why the repository has a `Dockerfile` but no registry
configuration, no image reference and no build workflow: RunPod builds it.

RunPod builds automatically on every push to `main` — the console's **Builds**
tab carries one entry per commit, and no manual trigger is needed. What a
build does not do is announce itself: a completed build plus a stale version
reading looks identical to a failed deployment. A release is only real once
the live worker reports the expected identity through `system_info`.

## Deploying a release

1. Merge the release to `main` and confirm `app/system_v2.py` declares the
   intended `VERSION` and `BUILD`.
2. Watch the RunPod console's **Builds** tab. The push starts a build on its
   own; wait for it to complete and for workers to roll over. Only trigger a
   build by hand if none appeared.
3. Verify, by either route:
   - the **Verify StemForge Deployment** workflow
     (`.github/workflows/verify-stemforge-deployment.yml`) from the Actions
     tab, which also runs the test suite first; or
   - the console's **Requests** tab, posting
     `{"input":{"action":"system_info"}}` and reading `stemforge_version` and
     `build` directly. This is free and needs no merge.

Re-check after the build completes, not before. A version reading taken while
a build is still running is the single easiest way to conclude a release
failed when it is minutes from being live.

The verify workflow compiles the worker, runs the full pytest suite, validates
request files, asserts the source declares the expected version and build, then
polls the live endpoint until it reports that exact identity. It also asserts
the Naturalize and quality-gate contract:

- `naturalize.default_intensity == 1.0`
- `naturalize.preset == "Naturalize Balanced"`
- `naturalize.maximum_full_passes == 2`
- `naturalize.second_pass_maximum_intensity_scale == 0.70`
- `naturalize.denoise_constraints.position == "post_cocktail_only"`
- `quality_gates.automatic_intensity_reduction == true`
- `quality_gates.vocoder_fallback_to_dsp == true`
- `reliability_guards.lazy_optional_vocoder_imports == true`

Any mismatch fails the run. It reports live backend availability (Vocos,
BigVGAN, DisCoder, torch, demucs, whisperx) and uploads the live `system_info`
as a 90-day artifact. Do not submit paid audio work until it passes.

## Rolling back

Rollback is the same operation in reverse: trigger a build in the RunPod
console from the previous known-good commit or tag, then re-run the verify
workflow with `expected_version` and `expected_build` set to that release.
Record the last known-good commit in the table below so it is always at hand.

## API scope note

Two different RunPod scopes are involved, and the repository only holds one.

| API | Used for | Status |
| --- | --- | --- |
| `api.runpod.ai/v2/{endpoint}/run` and `/status` | submitting jobs, `system_info` verification | works with the stored `RUNPOD_API_KEY` |
| `rest.runpod.io/v1/endpoints`, `/templates` | reading or changing endpoint and template config | **401** with the stored key |

`probe-live-stemforge.yml` and `probe-runpod-template.yml` query
`rest.runpod.io/v1` and fail for this reason. The verify workflow deliberately
uses only the `v2` job API. Adding a template-scoped key as a separate secret
would let automation read the deployment configuration, but it is not needed to
deploy or verify a GitHub-source endpoint.

## Release record

| Field | v2.4.0 |
| --- | --- |
| Status | **Deployed and verified live** |
| Source commit | `806daa6` (tag `v2.4.0`), contained in `main` |
| Build method | RunPod GitHub source, branch `main`, automatic build on push |
| Active build | commit `4e8408f` ("Refresh live probe status"), completed 2026-08-04 17:40 local, 18/18 layers, 6.27 GB pushed |
| Endpoint id | `dyup1dztjr4u15` |
| Verification date | 2026-08-04 |
| Verification job id | `9f0e8aa1-29b4-43f7-b6b9-00011419adc1-u1` |
| Worker id | `f54z1ziuuzqiry` |
| Live version / build | `2.4.0` / `v2.4.0-naturalize-balanced-vocoder` |
| Contract assertions | all pass: default_intensity 1.0, preset Naturalize Balanced, maximum_full_passes 2, second_pass scale 0.70, denoise post_cocktail_only, automatic_intensity_reduction, vocoder_fallback_to_dsp, lazy_optional_vocoder_imports |
| Backend availability | Vocos **available** (`charactr/vocos-mel-24khz`); BigVGAN unavailable (modules absent); DisCoder unavailable (env not configured) |
| Cold start | 12.5 s delay, 194 ms execution, no startup failure |
| Rollback target | the preceding entry in the RunPod Builds history |

Note: `storage.bucket_configured` is `false`, so outputs persist to the RunPod
volume rather than signed S3 links. That is pre-existing configuration, not a
v2.4.0 regression.

Fill this in from live `system_info` after each release. Never record a
deployment as complete unless live verification confirmed the exact identity.
