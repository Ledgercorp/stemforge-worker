# StemForge relay contract

Every live render in this repository starts the same way: a JSON request file
is committed to `main`, a workflow picks it up, POSTs it to the RunPod
endpoint, polls until the job reaches a terminal state, and commits the result
back. Nothing validates the file between "committed" and "a GPU worker is
running", so a typo in `input.action` or a missing audio source costs a full
round trip.

This document is the contract that request files must satisfy.
`tools/validate_jobs.py` enforces it mechanically:

```bash
python tools/validate_jobs.py                     # every request file
python tools/validate_jobs.py jobs/my-job.json    # one file, before committing
python tools/validate_jobs.py --strict            # warnings fail too
python tools/validate_jobs.py --format json       # machine-readable
```

Errors mean the job will be rejected or will waste a worker. Warnings mean it
will probably run but something about it is fragile.

## Request envelope

```json
{
  "artifact_name": "Optional_Release_Asset_Name",
  "input": {
    "action": "system_info"
  },
  "policy": {
    "ttl": 600000,
    "executionTimeout": 300000
  }
}
```

- `input` (required) is the payload passed verbatim to `handler.py`.
- `input.action` (required) must be one of the actions in `app/actions_v2.py`.
- `policy` is RunPod's, in **milliseconds**. `executionTimeout` is the run
  budget, `ttl` is how long the job may live in total, so `executionTimeout`
  must never exceed `ttl`.
- `artifact_name`, `notes` and `requested_by` are read by workflows and
  humans, not by the worker. Any other top-level key is ignored by RunPod and
  is almost always a mistake — most often a payload written as `inputs`.

## Getting your music to the worker

S3 is not configured (`storage.bucket_configured` is `false`) and this
endpoint has **no shared network volume**, so there are two reliable ways in:

| Situation | Use |
| --- | --- |
| Anything of real length | `audio_url` - any stable HTTPS source |
| A few seconds of audio | `audio_base64` inline, under ~1 MB encoded |

Google Drive links work but are fragile: they expire and can serve an HTML
quota page instead of audio. Prefer a stable host where you have one.

Uploading to the volume is **not** a third option here. See the caveat below.

### Direct upload

`tools/upload_audio.py` pushes a local file straight to the RunPod volume
using the worker's own `volume_upload_*` actions. It needs only a run-scoped
key — the same one the relay uses:

```bash
export RUNPOD_API_KEY=...
python tools/upload_audio.py ~/Music/mysong.wav
```

It prints the volume path. Chunks are 5 MB (the worker's ceiling is 7 MB),
and size plus SHA-256 are verified on both ends — a mismatch deletes the
assembled file rather than leaving a corrupt input in place.

That path then goes into any job as `audio_path`, `master_path`, or a stem's
`path`:

```json
{"input": {"action": "naturalize", "mode": "auto",
           "audio_path": "/runpod-volume/stemforge/transfers/mysong-9f0e8aa1/mysong.wav"}}
```

To go straight from a file to a submittable request:

```bash
python tools/upload_audio.py ~/Music/mysong.wav \
  --emit-job jobs/mysong-naturalize-20260805-1200.json \
  --action naturalize --mode auto --song "My Song"
```

Uploading several stems at once works too — pass multiple files and each
prints its path, in order.

**This does not work on the current endpoint.** Each chunk is a separate job
and may land on a different worker, and `/runpod-volume` is container-local
here, so the chunks never accumulate in one place and a later render cannot
read the assembled file. The tool and the worker actions are correct; the
prerequisite is a RunPod **network volume** attached to the endpoint. Attach
one and this becomes the best path for large inputs, since a single upload
then serves every later render. Until then, use a URL.

## Supplying audio

Every file input follows the same four-suffix convention, resolved by
`materialize_input` in `app/storage.py`:

| Form | Use |
| --- | --- |
| `<field>_url` | any HTTPS source |
| `<field>_storage_key` | S3-compatible key created by `create_upload` |
| `<field>_base64` | small inline payloads only |
| `<field>_path` | a path already inside the RunPod volume |

An object (`{"url": ..., "sha256": ..., "size_bytes": ...}`) or a bare HTTPS
string under `<field>` works too. `<field>_sha256` and `<field>_size_bytes`
enable integrity verification on download — worth setting for anything long.

### Required inputs per action

| Action | Required |
| --- | --- |
| `analyze_audio_v2`, `master_audio`, `master_audio_github_delivery`, `repair_audio`, `humanize_audio`, `separate_stems`, `naturalize` | `audio` |
| `compare_audio_v2` | `source` and `candidate` |
| `stem_remix` | `master` + at least one stem |
| `full_pass` | `master` (stems optional; a master-only pass is legal) |
| `inspect_stems_v2` | at least one stem (`master` optional) |
| `render_lyric_video` | `audio`, `background`, and `subtitles` or `lines` |
| `align_lyrics`, `align_lyrics_smart` | `lyrics` + one of `vocal_url`, `audio_url`, `audio_base64` |
| `analyze_audio` (legacy) | `audio_url` |
| `compare_audio` (legacy) | `source_url`, `candidate_url` |
| `create_download` | `storage_key` |
| `delete_storage_objects` | `storage_keys` |
| `record_rule` | `rule` |
| `record_feedback` | `feedback` |

Stems may be an array of objects (`{"name": ..., "url": ...}`), an array of
URL strings, or a name→URL object. Give every stem an explicit name: names
drive role routing, gain staging and reconstruction reporting. Auxiliary
parts that must not participate in reconstruction fitting use
`"role": "bass_pad"` or `"overlay": true`.

### Vocabularies

Values outside these sets are rejected by the worker, so the validator treats
them as errors:

| Field | Values |
| --- | --- |
| `mix_profiles` (`stem_remix`) | `balanced`, `heavy` |
| `profiles` (`master_audio`, `full_pass`) | `dynamic`, `balanced`, `dense` |
| `remix_mode` | `auto`, `full_stem_mix`, `master_anchored_delta` |
| `mode` (`naturalize`) | `auto`, `quick`, `surgical` |
| `vocoder` / `vocoder.backend` | `auto`, `off`, `vocos`, `bigvgan`, `discoder` |

Naturalize numeric policy: `intensity` is `0` (true bypass) or `0.6`–`1.4`
(values between 2 and 140 are read as percentages), `passes` is 1 or 2, and
`second_pass_intensity` is capped at `0.70`. The worker clamps rather than
rejects, so the validator warns — but a clamped render is not the render that
was asked for.

## Policy budgets

`full_pass`, `stem_remix`, `naturalize`, `master_audio`, `separate_stems`,
`render_lyric_video` and the lyric-alignment actions routinely run for many
minutes on a cold GPU worker. Set an explicit budget; without one the endpoint
default applies and long renders die partway through:

```json
"policy": { "executionTimeout": 5400000, "ttl": 7200000 }
```

`system_info` and other metadata actions are fine at `300000` / `600000`.

## Naming and one job per commit

Name request files `<slug>-YYYYMMDD-HHMM.json`, kebab-case, e.g.
`hypervigilant-bass-pad-remix-20260804-1252.json`. The result lands at the
same stem in the matching results directory, so the timestamp is what makes a
rerun distinguishable from the run it replaces.

The relay submits **every** job file changed in a push. Commit one request at
a time unless a batch is genuinely intended.

## Getting rendered audio back

This endpoint has **no shared network volume**: `/runpod-volume` is
container-local and dies with the worker. A render's own report will list
volume paths and they are real, but a later job runs on a different worker and
cannot read them - `volume_file_info` returns `FileNotFoundError` minutes
later. Do not plan on fetching output after the fact.

Instead, put the request in `delivery_requests/` and let
`run-render-delivered.yml` handle it. The worker uploads to a temporary
private release while it is still alive; the workflow collects the assets into
a 90-day artifact and deletes the release. The release token is injected at run
time and never appears in a committed file.

Supported actions: `full_pass`, `naturalize`, `stem_remix`, `master_audio`,
`master_audio_github_delivery`.

Download from the workflow run's **Artifacts** section. The run summary also
carries the naturalize verdict, safety metrics and a per-profile candidate
table.

Configuring `STEMFORGE_S3_BUCKET` and S3 credentials would remove the need for
this: every job would return signed download links directly. Until then,
delivery is the only path that survives worker shutdown.

## Where requests go

| Directory | Workflow | Purpose |
| --- | --- | --- |
| `jobs/` | `stemforge-relay.yml` | general-purpose submit-and-poll |
| `system_probe_requests/` | `probe-live-stemforge.yml` | live version/build probes |
| `stem_remix_jobs/`, `stem_remix_retry_jobs/` | `run-stemforge-stem-remix.yml` | remix renders with release delivery |
| `full_pass_jobs/`, `full_pass_retry_jobs/` | `run-stemforge-full-pass-export*.yml` | autonomous full pass |
| `full_pass_direct_requests/` | `stemforge-full-pass-direct-fetch.yml` | full pass with direct fetch |
| `v2_export_jobs/` | `run-stemforge-master-export.yml` | master export |
| `v2_jobs/` | `stemforge-v2-verify.yml` | v2 verification runs |
| `v201_quick_checks/`, `v201_quick_jobs/` | `stemforge-v201-quick-check.yml` | fast liveness checks |
| `direct_master_requests/`, `direct_balanced_requests/` | `stemforge-direct-*-fetch.yml` | direct candidate fetches |

Result directories (`results/`, `probe_results/`, `system_probe_results/`, …)
are written by workflows. Never hand-edit them: they are the audit trail for
what the live endpoint actually returned.

## Reading results

`results/<name>.submit.json` is the submission response; `results/<name>.json`
is the last polled status. Terminal states and what they mean:

| Status | Meaning | Next step |
| --- | --- | --- |
| `COMPLETED` | worker finished — check `output.status`, which may still be `rejected` | read the report before claiming success |
| `FAILED` | worker raised | read `output.error` / `error_type` |
| `TIMED_OUT` | exceeded `executionTimeout` | raise the budget, or split the work |
| `CANCELLED` | cancelled via `cancel_requests/` | — |
| `REJECTED` | the relay refused the file before submission | fix the file; the validator catches these locally |
| `SUBMIT_FAILED` | RunPod refused the POST | check endpoint id, secret, payload size |
| `STATUS_HTTP_ERROR` | polling failed | usually transient |
| `RELAY_TIMEOUT` | 25 minutes of polling without a terminal state | the job may still be running; probe it by id |

A `COMPLETED` relay status is not a successful render. The worker returns
`{"status": "rejected", "reason": ...}` inside `output` whenever a quality
gate refuses a candidate, and that is a normal, intended outcome.

## Hard rules

1. **Never commit a credential.** `github_release_export.token` and anything
   named `token`, `secret`, `password`, `api_key` or `access_key` must be
   injected by the workflow from a repository secret. The validator fails on
   literal values and on GitHub/RunPod/AWS key shapes anywhere in the file.
2. **Verify the live version before trusting a render.** A merged release does
   not mean the endpoint rolled forward. Probe `system_info` and check
   `stemforge_version` and `build`. Long-running request files can pin
   `input.minimum_worker_version` and `input.required_build`, which the remix
   workflow enforces before it spends GPU time.
3. **Prefer storage keys or release assets over Google Drive links.** Drive
   serves an HTML quota page instead of audio under load, which surfaces much
   later as a confusing decode failure.
4. **Keep inline base64 small.** Anything approaching a megabyte belongs in
   storage; committed base64 lives in git history forever.
5. **Don't edit results.** Reruns get a new request file with a new timestamp.
