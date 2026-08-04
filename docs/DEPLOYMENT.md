# StemForge deployment

The RunPod Serverless endpoint `dyup1dztjr4u15` builds its container image from
this repository's **GitHub source on `main`**. There is no external container
registry, which is why the repository has a `Dockerfile` but no registry
configuration, no image reference and no build workflow: RunPod builds it.

Automatic deploy-on-push is **off**. Merging to `main` and tagging a release
therefore change nothing on the endpoint by themselves — a build has to be
started in the RunPod console. A release is only real once the live worker
reports the expected identity through `system_info`.

## Deploying a release

1. Merge the release to `main` and confirm `app/system_v2.py` declares the
   intended `VERSION` and `BUILD`.
2. In the RunPod console, open endpoint `dyup1dztjr4u15` and trigger a new
   build from the GitHub source. It builds the current `main`.
3. Wait for the build to finish and the workers to roll over.
4. Run the **Verify StemForge Deployment** workflow
   (`.github/workflows/verify-stemforge-deployment.yml`) from the Actions tab.

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
| Status | **Not deployed** — build not yet triggered |
| Source commit | `806daa6` (tag `v2.4.0`), contained in `main` |
| Local validation | compile OK; 31 tests pass; local `system_info` matches 2.4.0 / v2.4.0-naturalize-balanced-vocoder; local naturalize smoke test accepted with gates unmodified |
| Build method | RunPod GitHub source, branch `main`, auto-deploy off |
| Last successful build | predates the v2.4.0 merge |
| Endpoint id | `dyup1dztjr4u15` |
| Deployment timestamp | — |
| Verification job id | — |
| Live version / build | `2.3.0` / `v2.3.0-naturalize-quality` |
| Backend availability | — not observed live |
| Rollback target | the commit behind the last successful build (v2.3.0) |

Fill this in from the verify workflow's artifact after each release. Never
record a deployment as complete unless live `system_info` verified the exact
identity.
