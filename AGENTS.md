# Working agreement for agents on this repository

Two agents operate StemForge, from opposite ends of the same pipeline:

- **The StemForge GPT** is the live operator. It talks to the artist, decides
  what to render, writes request files, watches results, and judges output.
- **Claude Code** is the engine side. It changes `app/`, `tests/`,
  `.github/workflows/` and the docs, on a branch, with CI green.

Both write to the same `main`, and neither sees the other's session. This file
is the shared contract so the two halves compose instead of colliding.

Read `docs/RELAY_CONTRACT.md` before writing or reviewing any request file.

## Ownership

| Path | Written by | Rule |
| --- | --- | --- |
| `jobs/`, `*_requests/`, `*_jobs/`, `*_checks/` | GPT | one request per commit; validate first |
| `results/`, `*_results/`, `probe_results/` | workflows | never hand-edited by either agent |
| `app/`, `handler.py`, `tests/`, `requirements*.txt`, `Dockerfile` | Claude | branch + CI, never committed straight to `main` |
| `.github/workflows/` | Claude | GPT may trigger them, not rewrite them |
| `release/`, `release-trigger/` | either | only after the code being released is merged |
| `README.md`, `V2_FEATURES.md`, `docs/` | Claude | must describe what is deployed, not what is planned |

## Ground rules

These hold for both agents, every time.

1. **The live endpoint is the source of truth, not `main`.** A merged release
   does not mean the endpoint rolled forward. Probe `system_info` and check
   `stemforge_version` and `build` before trusting or reporting a render.
2. **`COMPLETED` is not success.** The relay reports the RunPod job state. The
   worker's own verdict is `output.status`, and `{"status": "rejected"}` from
   a quality gate is a normal outcome. Read the report before claiming a
   render worked.
3. **Never commit a credential.** Tokens come from repository secrets, in the
   workflow. `tools/validate_jobs.py` fails on literal ones.
4. **Never hand-edit a result.** Reruns get a new request file with a new
   timestamp. The result directories are the audit trail of what the endpoint
   actually returned.
5. **Say what is estimated.** Demucs stems are estimates, objective comparison
   is a DSP proxy, and mastering candidates still need level-matched
   listening. Neither agent should describe them otherwise.

## Before the GPT submits a request

```bash
python tools/validate_jobs.py <path-to-request>.json
```

- Action exists in `app/actions_v2.py` **and** in the deployed build.
- Every required input is present, in one of the four accepted forms.
- `policy.executionTimeout` is set for anything long, and is `<= policy.ttl`.
- Filename is `<slug>-YYYYMMDD-HHMM.json`.
- No credentials, no oversized inline base64.

The validator runs in CI on every push and pull request that touches a request
directory, so a bad file is caught even if the local run is skipped. It is a
contract check, not a network check: it cannot tell whether a URL still
resolves or whether the deployed build supports a newly added action.

## Before Claude merges an engine change

- `python -m pytest -q` passes, and new behavior has a test.
- `python tools/validate_jobs.py` still reports zero errors — a change to the
  action list, a vocabulary or an input requirement must land together with
  the validator update, or `tests/test_validate_jobs.py` fails on drift.
- Any new action, profile or mode is added to `docs/RELAY_CONTRACT.md` in the
  same change, so the GPT can use it without guessing.
- `README.md` and `V2_FEATURES.md` describe the deployed build.

## Handing work across

The durable channel between the two agents is this repository, not chat
history. When the GPT hits an engine limitation, it should record the failing
request path and the exact `output` verdict — a GitHub issue, or a note in the
handoff commit — so the fix can start from the real payload rather than a
paraphrase. When Claude changes behavior the GPT depends on, the contract doc
and the release notes are how the GPT finds out.

Version drift is the failure mode that has cost the most time here: a request
using an action the live build does not have comes back `COMPLETED` with
`{"status": "rejected", "reason": "Unsupported action: ..."}`. Long-running
request files can pin `input.minimum_worker_version` and
`input.required_build`; the remix workflow enforces both before spending GPU
time.
