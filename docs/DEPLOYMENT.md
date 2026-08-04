# StemForge deployment

The worker runs on a RunPod Serverless endpoint. A GitHub release, a merged
pull request and a git tag change nothing on their own: the endpoint runs
whatever container image its template points at, so a deployment is only real
once the live worker reports the expected identity through `system_info`.

## Prerequisites

Deployment needs three things this repository does not currently carry.

| Requirement | Secret | Notes |
| --- | --- | --- |
| RunPod key with template scope | `RUNPOD_API_KEY_ADMIN` | The existing `RUNPOD_API_KEY` is run-scoped: it works against `api.runpod.ai/v2/{endpoint}/run` but returns **401** from `rest.runpod.io/v1`, so it cannot read or update the template. |
| Container registry account | `REGISTRY_USERNAME`, `REGISTRY_PASSWORD` | No registry has ever been configured here. Every image on the endpoint so far was built and pushed outside this repository. |
| Image repository name | workflow input | Supplied per run as `image_repository`; it is not stored in the repo, and must not be guessed. |

## Running a deployment

`.github/workflows/deploy-stemforge.yml`, by manual `workflow_dispatch` or a
`v*` tag push. In order it:

1. compiles the worker, runs the full pytest suite, validates request files,
   and asserts `app/system_v2.py` declares the expected version and build;
2. builds from the checked-out commit and pushes two immutable tags —
   `:<version>` and `:sha-<commit>` — recording the digest;
3. reads the endpoint and its bound template, and writes a rollback snapshot
   (endpoint id, template id, previous image, scaling, GPU selection,
   container disk, volume config and mount path, timeouts, entrypoint and
   start command, plus environment variable **names** only);
4. `PATCH`es only `imageName` on the existing template, which triggers
   RunPod's rolling release. No other setting and no environment variable is
   touched, and no replacement endpoint is created;
5. polls `system_info` until the live worker reports the expected identity, and
   asserts the full Naturalize and quality-gate contract. A mismatch fails the
   run;
6. uploads the rollback snapshot and the live `system_info` as a 90-day
   artifact.

Roll back by re-running with `rollback_image` set to the previous reference
from the snapshot. That skips the build and points the template back.

Pin by digest (`repository@sha256:...`, the default) rather than a mutable tag,
so a redeployed tag can never silently change what the endpoint runs.

## Release record

| Field | v2.4.0 |
| --- | --- |
| Status | **Not deployed** |
| Source commit | `806daa6` (tag `v2.4.0`); `main` at `7131d50` |
| Local validation | compile OK; 31 tests pass; local `system_info` matches 2.4.0 / v2.4.0-naturalize-balanced-vocoder; local naturalize smoke test accepted |
| Image tag | — not built (no registry configured) |
| Image digest | — |
| RunPod template id | — not retrievable; `rest.runpod.io/v1` returns 401 with the stored key |
| Endpoint id | `dyup1dztjr4u15` |
| Deployment timestamp | — |
| Verification job id | — |
| Live version / build | `2.3.0` / `v2.3.0-naturalize-quality` (unchanged) |
| Backend availability | — not observed live |
| Rollback image | — unknown; the current image name has never been successfully read |

Keep this table updated per release, filling it from the workflow's deployment
artifact rather than from memory. Do not record a deployment as complete unless
live `system_info` verified the exact identity.
