#!/usr/bin/env python3
"""Validate StemForge relay job files before they reach RunPod.

Every live render starts as a JSON file committed to one of the request
directories in this repository. A workflow picks the file up and POSTs it
straight to the RunPod endpoint, so a typo in ``input.action`` or a missing
audio source is only discovered after a GPU worker has already spun up.

This validator applies the same rules the worker applies, locally and in CI:

    python tools/validate_jobs.py                    # every request file
    python tools/validate_jobs.py jobs/my-job.json   # one file
    python tools/validate_jobs.py --strict           # warnings fail too
    python tools/validate_jobs.py --format json      # machine-readable

Exit status is 0 when the checked files are submittable, 1 when they are not.

The supported action list comes from ``app/actions_v2.py`` so it can never
drift from the deployed worker. That module is loaded straight from its file
rather than imported as ``app.actions_v2``, because the ``app`` package pulls
in numpy at import time and this script has to run on a bare runner with
nothing but the standard library installed.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_supported_actions() -> frozenset[str]:
    """Read SUPPORTED_ACTIONS from the worker without importing the package."""
    module_path = REPO_ROOT / "app" / "actions_v2.py"
    spec = importlib.util.spec_from_file_location("stemforge_actions_v2", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Cannot load action list from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return frozenset(module.SUPPORTED_ACTIONS)


SUPPORTED_ACTIONS = _load_supported_actions()

# Directories a workflow watches for request files. Files without a top-level
# "input" object are workflow control files (run ids, fetch targets) and are
# skipped by discovery.
REQUEST_DIRECTORIES = (
    "cancel_requests",
    "direct_balanced_requests",
    "direct_master_requests",
    "download_requests",
    "export_requests",
    "full_pass_direct_requests",
    "full_pass_jobs",
    "full_pass_retry_jobs",
    "jobs",
    "maintenance",
    "status_probe_jobs",
    "stem_remix_jobs",
    "stem_remix_retry_jobs",
    "system_probe_requests",
    "v2_balanced_fetch",
    "v2_bridge_requests",
    "v2_export_jobs",
    "v2_jobs",
    "v2_output_requests",
    "v201_quick_checks",
    "v201_quick_jobs",
)

# An input file may be supplied as <field>_url, <field>_storage_key,
# <field>_base64, <field>_path, or as a <field> object/URL string.
SOURCE_SUFFIXES = ("url", "storage_key", "base64", "path")

# Audio inputs each action materializes as required (app/storage.py).
REQUIRED_SOURCES: dict[str, tuple[str, ...]] = {
    "analyze_audio_v2": ("audio",),
    "compare_audio_v2": ("source", "candidate"),
    "full_pass": ("master",),
    "humanize_audio": ("audio",),
    "master_audio": ("audio",),
    "master_audio_github_delivery": ("audio",),
    "naturalize": ("audio",),
    "render_lyric_video": ("audio", "background"),
    "repair_audio": ("audio",),
    "separate_stems": ("audio",),
    "stem_remix": ("master",),
}

# Actions the worker rejects outright without at least one stem.
REQUIRES_STEMS = ("inspect_stems_v2", "stem_remix")

# Plain scalar fields the worker rejects the job without.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "analyze_audio": ("audio_url",),
    "compare_audio": ("source_url", "candidate_url"),
    "create_download": ("storage_key",),
    "delete_storage_objects": ("storage_keys",),
    "record_feedback": ("feedback",),
    "record_rule": ("rule",),
}

# Groups where any one member satisfies the requirement.
REQUIRED_ANY: dict[str, tuple[tuple[str, ...], ...]] = {
    "align_lyrics": (("vocal_url", "audio_url", "audio_base64"),),
    "align_lyrics_smart": (("vocal_url", "audio_url", "audio_base64"),),
}
for _lyric_action in ("align_lyrics", "align_lyrics_smart"):
    REQUIRED_FIELDS[_lyric_action] = ("lyrics",)

# Vocabularies mirrored from the worker. tests/test_validate_jobs.py asserts
# these stay in sync with app/stem_remix_v2.py and app/mastering_v2.py, so the
# validator itself never has to import numpy.
MIX_PROFILES = ("balanced", "heavy")
MASTER_PROFILES = ("dynamic", "balanced", "dense")
REMIX_MODES = ("auto", "full_stem_mix", "master_anchored_delta")
NATURALIZE_MODES = ("auto", "quick", "surgical")
VOCODER_BACKENDS = ("auto", "off", "vocos", "bigvgan", "discoder")

# Actions that routinely run for many minutes on a GPU worker. RunPod applies
# its endpoint default when policy.executionTimeout is absent, which has
# silently killed long renders before.
LONG_RUNNING_ACTIONS = (
    "align_lyrics",
    "align_lyrics_smart",
    "full_pass",
    "master_audio",
    "master_audio_github_delivery",
    "naturalize",
    "render_lyric_video",
    "separate_stems",
    "stem_remix",
)
LONG_RUNNING_EXECUTION_TIMEOUT_MS = 1_800_000

ALLOWED_TOP_LEVEL_KEYS = (
    "artifact_name",
    "input",
    "notes",
    "policy",
    "requested_by",
    "webhook",
)

# Credential shapes that must never be committed to a job file.
SECRET_PATTERNS = (
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), "GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(r"\brpa_[A-Za-z0-9]{16,}"), "RunPod API key"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{20,}"), "bearer token"),
)
SECRET_FIELD_NAMES = ("token", "secret", "password", "api_key", "access_key")
PLACEHOLDER_PATTERN = re.compile(r"^\s*(\$\{\{.*\}\}|\$[A-Z_]+|<.*>|\*+|REDACTED)\s*$")

FILENAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d{8}-\d{4}$")

INLINE_BYTES_WARNING = 1_048_576

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One validation finding against one job file."""

    path: Path
    level: str
    message: str

    def render(self) -> str:
        relative = _relative(self.path)
        return f"{relative}: {self.level}: {self.message}"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _has_source(payload: dict, field: str) -> bool:
    """True when an input file is supplied for ``field`` in any accepted form."""
    spec = payload.get(field)
    if isinstance(spec, str) and spec.strip():
        return True
    if isinstance(spec, dict) and any(
        spec.get(key) for key in ("url", "storage_key", "base64", "path")
    ):
        return True
    return any(payload.get(f"{field}_{suffix}") for suffix in SOURCE_SUFFIXES)


def _stem_entries(payload: dict) -> list[dict]:
    """Normalize the stem shapes the worker accepts into a list of dicts."""
    stems = payload.get("stems")
    if stems is None:
        stems = payload.get("stem_urls")
    if stems is None:
        return []
    if isinstance(stems, dict):
        return [{"name": name, "url": url} for name, url in stems.items()]
    if not isinstance(stems, list):
        return []
    entries: list[dict] = []
    for index, item in enumerate(stems):
        if isinstance(item, str):
            entries.append({"name": f"stem_{index + 1}", "url": item})
        elif isinstance(item, dict):
            entries.append(item)
        else:
            entries.append({})
    return entries


def _walk(value: Any, trail: str = "") -> Iterable[tuple[str, Any]]:
    """Yield (dotted path, value) for every scalar in a JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{trail}.{key}" if trail else str(key)
            yield from _walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{trail}[{index}]")
    else:
        yield trail, value


def _suggest(action: str) -> str:
    matches = difflib.get_close_matches(action, sorted(SUPPORTED_ACTIONS), n=1)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def _check_secrets(path: Path, document: Any) -> list[Issue]:
    issues: list[Issue] = []
    for trail, value in _walk(document):
        if not isinstance(value, str) or not value.strip():
            continue
        if PLACEHOLDER_PATTERN.match(value):
            continue
        leaf = trail.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if leaf in SECRET_FIELD_NAMES:
            issues.append(
                Issue(
                    path,
                    ERROR,
                    f"{trail} contains a literal credential. Inject it from a "
                    "repository secret in the workflow instead of committing it.",
                )
            )
            continue
        # base64 payloads are large and can incidentally match token shapes.
        if leaf.endswith("base64"):
            continue
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(value):
                issues.append(
                    Issue(path, ERROR, f"{trail} looks like a committed {label}.")
                )
                break
    return issues


def _check_policy(path: Path, document: dict, action: str) -> list[Issue]:
    issues: list[Issue] = []
    policy = document.get("policy")
    if policy is None:
        if action in LONG_RUNNING_ACTIONS:
            issues.append(
                Issue(
                    path,
                    WARNING,
                    f"'{action}' is long-running but no policy is set. Add "
                    "policy.executionTimeout and policy.ttl so the endpoint "
                    "default cannot cut the render short.",
                )
            )
        return issues

    if not isinstance(policy, dict):
        return [Issue(path, ERROR, "policy must be an object.")]

    numbers: dict[str, int] = {}
    for key in ("ttl", "executionTimeout"):
        if key not in policy:
            continue
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(
                Issue(path, ERROR, f"policy.{key} must be an integer in milliseconds.")
            )
            continue
        if value <= 0:
            issues.append(Issue(path, ERROR, f"policy.{key} must be positive."))
            continue
        numbers[key] = value

    ttl = numbers.get("ttl")
    execution_timeout = numbers.get("executionTimeout")
    if ttl is not None and execution_timeout is not None and execution_timeout > ttl:
        issues.append(
            Issue(
                path,
                ERROR,
                f"policy.executionTimeout ({execution_timeout} ms) exceeds "
                f"policy.ttl ({ttl} ms); the job would be discarded before it "
                "can finish.",
            )
        )

    if action in LONG_RUNNING_ACTIONS:
        if execution_timeout is None:
            issues.append(
                Issue(
                    path,
                    WARNING,
                    f"'{action}' is long-running but policy.executionTimeout is "
                    "not set; the endpoint default applies.",
                )
            )
        elif execution_timeout < LONG_RUNNING_EXECUTION_TIMEOUT_MS:
            issues.append(
                Issue(
                    path,
                    WARNING,
                    f"policy.executionTimeout is {execution_timeout} ms for "
                    f"'{action}'; renders of this kind regularly need more than "
                    f"{LONG_RUNNING_EXECUTION_TIMEOUT_MS} ms.",
                )
            )
    return issues


def _check_naturalize(path: Path, payload: dict) -> list[Issue]:
    issues: list[Issue] = []
    mode = payload.get("mode")
    if mode is not None and str(mode).lower() not in NATURALIZE_MODES:
        issues.append(
            Issue(
                path,
                ERROR,
                f"mode '{mode}' is not one of {', '.join(NATURALIZE_MODES)}.",
            )
        )

    if "intensity" in payload:
        intensity = payload["intensity"]
        if isinstance(intensity, bool) or not isinstance(intensity, (int, float)):
            issues.append(Issue(path, ERROR, "intensity must be a number."))
        else:
            scaled = intensity / 100.0 if 2.0 < intensity <= 140.0 else float(intensity)
            if scaled != 0.0 and not 0.6 <= scaled <= 1.4:
                issues.append(
                    Issue(
                        path,
                        WARNING,
                        f"intensity {intensity} is outside the 0.6-1.4 policy "
                        "range and will be clamped by the worker.",
                    )
                )

    if "passes" in payload:
        passes = payload["passes"]
        if isinstance(passes, bool) or not isinstance(passes, int):
            issues.append(Issue(path, ERROR, "passes must be an integer."))
        elif passes not in (1, 2):
            issues.append(
                Issue(
                    path,
                    WARNING,
                    f"passes {passes} will be clamped to the 1-2 range.",
                )
            )

    second = payload.get("second_pass_intensity", payload.get("second_pass_scale"))
    if second is not None:
        if isinstance(second, bool) or not isinstance(second, (int, float)):
            issues.append(Issue(path, ERROR, "second_pass_intensity must be a number."))
        elif not 0.0 <= second <= 0.70:
            issues.append(
                Issue(
                    path,
                    WARNING,
                    f"second_pass_intensity {second} will be clamped to 0.70.",
                )
            )

    vocoder = payload.get("vocoder", payload.get("vocoder_flag"))
    backend: Any = None
    if isinstance(vocoder, dict):
        backend = vocoder.get("backend")
    elif isinstance(vocoder, str):
        backend = vocoder
    if backend is not None and str(backend).lower() not in VOCODER_BACKENDS:
        issues.append(
            Issue(
                path,
                ERROR,
                f"vocoder backend '{backend}' is not one of "
                f"{', '.join(VOCODER_BACKENDS)}.",
            )
        )
    return issues


def _check_profiles(path: Path, payload: dict, action: str) -> list[Issue]:
    issues: list[Issue] = []
    if action == "stem_remix":
        remix_mode = payload.get("remix_mode")
        if remix_mode is not None and str(remix_mode).lower() not in REMIX_MODES:
            issues.append(
                Issue(
                    path,
                    ERROR,
                    f"remix_mode '{remix_mode}' is not one of "
                    f"{', '.join(REMIX_MODES)}.",
                )
            )
        requested = payload.get("mix_profiles")
        allowed: Sequence[str] = MIX_PROFILES
        label = "mix_profiles"
    elif action in ("master_audio", "master_audio_github_delivery", "full_pass"):
        requested = payload.get("profiles")
        allowed = MASTER_PROFILES
        label = "profiles"
    else:
        return issues

    if requested is None:
        return issues
    if isinstance(requested, str):
        requested = [requested]
    if not isinstance(requested, list):
        issues.append(Issue(path, ERROR, f"{label} must be a string or an array."))
        return issues

    unknown = [str(name) for name in requested if str(name) not in allowed]
    if unknown:
        issues.append(
            Issue(
                path,
                ERROR,
                f"unknown {label} {unknown}; available: {', '.join(allowed)}.",
            )
        )
    return issues


def _check_stems(path: Path, payload: dict, action: str) -> list[Issue]:
    issues: list[Issue] = []
    stems = payload.get("stems")
    if stems is not None and not isinstance(stems, (list, dict)):
        return [Issue(path, ERROR, "stems must be an array or an object.")]

    entries = _stem_entries(payload)
    if action in REQUIRES_STEMS and not entries:
        return [Issue(path, ERROR, f"'{action}' requires at least one stem.")]

    seen: set[str] = set()
    for index, entry in enumerate(entries):
        label = f"stems[{index}]"
        name = str(entry.get("name") or "").strip()
        if not name:
            issues.append(
                Issue(
                    path,
                    WARNING,
                    f"{label} has no name; the worker will call it "
                    f"'stem_{index + 1}' and role routing may be wrong.",
                )
            )
        elif name.lower() in seen:
            issues.append(
                Issue(path, ERROR, f"{label} repeats the stem name '{name}'.")
            )
        else:
            seen.add(name.lower())

        if not any(entry.get(key) for key in ("url", "storage_key", "base64", "path")):
            issues.append(
                Issue(
                    path,
                    ERROR,
                    f"{label} has no url, storage_key, base64 or path.",
                )
            )
    return issues


def _check_sources(path: Path, payload: dict, action: str) -> list[Issue]:
    issues: list[Issue] = []
    for field in REQUIRED_SOURCES.get(action, ()):
        if not _has_source(payload, field):
            issues.append(
                Issue(
                    path,
                    ERROR,
                    f"'{action}' requires {field}_url, {field}_storage_key, "
                    f"{field}_base64 or {field}_path.",
                )
            )

    for field in REQUIRED_FIELDS.get(action, ()):
        value = payload.get(field)
        empty = value is None or (isinstance(value, (str, list, dict)) and not value)
        if empty:
            issues.append(Issue(path, ERROR, f"'{action}' requires {field}."))

    for group in REQUIRED_ANY.get(action, ()):
        if not any(payload.get(field) for field in group):
            issues.append(
                Issue(
                    path,
                    ERROR,
                    f"'{action}' requires one of {', '.join(group)}.",
                )
            )

    if action == "render_lyric_video" and not (
        _has_source(payload, "subtitles") or payload.get("lines")
    ):
        issues.append(
            Issue(path, ERROR, "'render_lyric_video' requires subtitles or lines.")
        )
    return issues


def _check_hygiene(path: Path, document: dict, payload: dict) -> list[Issue]:
    issues: list[Issue] = []

    unknown_keys = sorted(set(document) - set(ALLOWED_TOP_LEVEL_KEYS))
    if unknown_keys:
        issues.append(
            Issue(
                path,
                WARNING,
                f"unrecognized top-level keys {unknown_keys}; RunPod only reads "
                "'input' and 'policy'.",
            )
        )

    if not FILENAME_PATTERN.match(path.stem):
        issues.append(
            Issue(
                path,
                WARNING,
                "filename should be kebab-case with a -YYYYMMDD-HHMM stamp so "
                "requests, results and reruns stay traceable.",
            )
        )

    drive_fields: list[str] = []
    for trail, value in _walk(payload):
        if not isinstance(value, str):
            continue
        if trail.rsplit(".", 1)[-1].split("[", 1)[0].lower().endswith("base64"):
            if len(value) > INLINE_BYTES_WARNING:
                issues.append(
                    Issue(
                        path,
                        WARNING,
                        f"{trail} inlines {len(value)} base64 characters into "
                        "git history; prefer a storage key or URL.",
                    )
                )
        elif "drive.google.com" in value:
            drive_fields.append(trail)

    if drive_fields:
        shown = ", ".join(drive_fields[:3])
        if len(drive_fields) > 3:
            shown += f", +{len(drive_fields) - 3} more"
        issues.append(
            Issue(
                path,
                WARNING,
                f"{len(drive_fields)} Google Drive link(s) ({shown}). Drive can "
                "serve an HTML quota page instead of audio; preflight them or "
                "prefer storage keys and release assets.",
            )
        )
    return issues


def validate_document(path: Path, document: Any) -> list[Issue]:
    """Validate one already-parsed job document."""
    if not isinstance(document, dict):
        return [Issue(path, ERROR, "job file must contain a JSON object.")]

    payload = document.get("input")
    if payload is None:
        return [Issue(path, ERROR, "job file must contain an 'input' object.")]
    if not isinstance(payload, dict):
        return [Issue(path, ERROR, "'input' must be an object.")]

    raw_action = payload.get("action")
    if raw_action is None:
        return [Issue(path, ERROR, "input.action is required.")]
    if not isinstance(raw_action, str):
        return [Issue(path, ERROR, "input.action must be a string.")]

    action = raw_action.strip()
    if action not in SUPPORTED_ACTIONS:
        return [
            Issue(
                path,
                ERROR,
                f"unsupported action '{action}'.{_suggest(action)} The deployed "
                "worker rejects it; confirm with system_info before retrying.",
            )
        ]

    issues: list[Issue] = []
    issues += _check_sources(path, payload, action)
    issues += _check_stems(path, payload, action)
    issues += _check_profiles(path, payload, action)
    if action == "naturalize":
        issues += _check_naturalize(path, payload)
    if action == "full_pass" and isinstance(payload.get("naturalize"), dict):
        issues += _check_naturalize(path, payload["naturalize"])
    issues += _check_policy(path, document, action)
    issues += _check_secrets(path, document)
    issues += _check_hygiene(path, document, payload)
    return issues


def validate_file(path: Path) -> list[Issue]:
    """Read and validate one job file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Issue(path, ERROR, f"cannot read file: {exc}")]
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return [Issue(path, ERROR, f"invalid JSON: {exc}")]
    return validate_document(path, document)


def discover(root: Path = REPO_ROOT) -> list[Path]:
    """Find every committed request file the relay workflows can submit."""
    found: list[Path] = []
    for directory in REQUEST_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        found.extend(sorted(base.rglob("*.json")))
    return found


def _looks_like_request(path: Path) -> bool:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(document, dict) and "input" in document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate StemForge relay job files before submission."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Job files to validate. Defaults to every request directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    args = parser.parse_args(argv)

    if args.paths:
        targets = [path for path in args.paths if path.suffix == ".json"]
        missing = [path for path in targets if not path.is_file()]
        if missing:
            for path in missing:
                print(f"{path}: error: file not found", file=sys.stderr)
            return 1
    else:
        targets = [path for path in discover() if _looks_like_request(path)]

    issues: list[Issue] = []
    for path in targets:
        issues.extend(validate_file(path))

    errors = [issue for issue in issues if issue.level == ERROR]
    warnings = [issue for issue in issues if issue.level == WARNING]

    if args.format == "json":
        print(
            json.dumps(
                {
                    "checked": len(targets),
                    "errors": len(errors),
                    "warnings": len(warnings),
                    "issues": [
                        {
                            "path": _relative(issue.path),
                            "level": issue.level,
                            "message": issue.message,
                        }
                        for issue in issues
                    ],
                },
                indent=2,
            )
        )
    else:
        for issue in errors + warnings:
            print(issue.render())
        print(
            f"Checked {len(targets)} job file(s): "
            f"{len(errors)} error(s), {len(warnings)} warning(s)."
        )

    if errors:
        return 1
    return 1 if (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
