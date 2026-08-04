"""Contract tests for the relay job validator.

The validator carries small copies of the worker's vocabularies so it can run
without numpy on a bare CI runner. These tests read the real worker modules
with ``ast`` (no import, no dependencies) and fail the build if a copy drifts.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tools import validate_jobs as validator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_tree(relative: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))


def _dict_keys(relative: str, name: str) -> set[str]:
    for node in ast.walk(_module_tree(relative)):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError(f"{name} not found in {relative}")


def _set_literals(relative: str) -> list[set[str]]:
    literals: list[set[str]] = []
    for node in ast.walk(_module_tree(relative)):
        if isinstance(node, ast.Set):
            values = {
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if values:
                literals.append(values)
    return literals


def _job(action: str, **payload: object) -> dict:
    return {"input": {"action": action, **payload}}


def _errors(document: dict, name: str = "probe-20260804-1200.json") -> list[str]:
    issues = validator.validate_document(Path(name), document)
    return [issue.message for issue in issues if issue.level == validator.ERROR]


def _warnings(document: dict, name: str = "probe-20260804-1200.json") -> list[str]:
    issues = validator.validate_document(Path(name), document)
    return [issue.message for issue in issues if issue.level == validator.WARNING]


def test_supported_actions_come_from_the_worker() -> None:
    worker_actions = _set_literals("app/actions_v2.py")
    assert any(
        actions == set(validator.SUPPORTED_ACTIONS) for actions in worker_actions
    ), "validator action list drifted from app/actions_v2.py"


def test_mix_profiles_match_the_worker() -> None:
    assert set(validator.MIX_PROFILES) == _dict_keys(
        "app/stem_remix_v2.py", "MIX_PROFILES"
    )


def test_master_profiles_match_the_worker() -> None:
    assert set(validator.MASTER_PROFILES) == _dict_keys(
        "app/mastering_v2.py", "PROFILES"
    )


def test_remix_modes_match_the_worker() -> None:
    assert set(validator.REMIX_MODES) in _set_literals("app/stem_remix_v2.py")


def test_vocoder_backends_match_the_worker() -> None:
    explicit = set(validator.VOCODER_BACKENDS) - {"auto", "off"}
    assert explicit in _set_literals("app/vocoder_v2.py")


def test_required_sources_only_reference_supported_actions() -> None:
    referenced = (
        set(validator.REQUIRED_SOURCES)
        | set(validator.REQUIRED_FIELDS)
        | set(validator.REQUIRED_ANY)
        | set(validator.REQUIRES_STEMS)
        | set(validator.LONG_RUNNING_ACTIONS)
    )
    assert referenced <= set(validator.SUPPORTED_ACTIONS)


def test_committed_request_files_are_submittable() -> None:
    """Every request file in the repository must still pass the contract."""
    failures: list[str] = []
    for path in validator.discover():
        if not validator._looks_like_request(path):
            continue
        failures.extend(
            issue.render()
            for issue in validator.validate_file(path)
            if issue.level == validator.ERROR
        )
    assert not failures, "\n".join(failures)


def test_unsupported_action_is_rejected_with_a_suggestion() -> None:
    messages = _errors(_job("naturalise", audio_url="https://example.com/a.wav"))
    assert len(messages) == 1
    assert "unsupported action 'naturalise'" in messages[0]
    assert "Did you mean 'naturalize'?" in messages[0]


def test_missing_action_is_rejected() -> None:
    assert _errors({"input": {}}) == ["input.action is required."]


def test_missing_input_is_rejected() -> None:
    assert _errors({"policy": {"ttl": 600000}}) == [
        "job file must contain an 'input' object."
    ]


@pytest.mark.parametrize(
    "action,field",
    [
        ("naturalize", "audio"),
        ("master_audio", "audio"),
        ("separate_stems", "audio"),
        ("full_pass", "master"),
    ],
)
def test_missing_audio_source_is_rejected(action: str, field: str) -> None:
    messages = _errors(_job(action))
    assert any(f"{field}_url" in message for message in messages)


def test_any_accepted_source_form_satisfies_the_requirement() -> None:
    for payload in (
        {"audio_url": "https://example.com/a.wav"},
        {"audio_storage_key": "incoming/a.wav"},
        {"audio_base64": "AAAA"},
        {"audio_path": "/runpod-volume/a.wav"},
        {"audio": {"storage_key": "incoming/a.wav"}},
        {"audio": "https://example.com/a.wav"},
    ):
        assert _errors(_job("naturalize", **payload)) == [], payload


def test_compare_audio_v2_requires_both_sides() -> None:
    messages = _errors(_job("compare_audio_v2", source_url="https://example.com/a.wav"))
    assert any("candidate_url" in message for message in messages)


def test_stem_remix_requires_stems() -> None:
    messages = _errors(_job("stem_remix", master_url="https://example.com/m.wav"))
    assert messages == ["'stem_remix' requires at least one stem."]


def test_stem_entries_need_a_source_and_unique_names() -> None:
    document = _job(
        "stem_remix",
        master_url="https://example.com/m.wav",
        stems=[
            {"name": "Drums", "url": "https://example.com/d.wav"},
            {"name": "Drums", "url": "https://example.com/d2.wav"},
            {"name": "Bass"},
        ],
    )
    messages = _errors(document)
    assert "stems[1] repeats the stem name 'Drums'." in messages
    assert "stems[2] has no url, storage_key, base64 or path." in messages


def test_unknown_mix_profile_is_rejected() -> None:
    document = _job(
        "stem_remix",
        master_url="https://example.com/m.wav",
        stems=["https://example.com/d.wav"],
        mix_profiles=["balanced", "punchy"],
    )
    assert any("unknown mix_profiles" in message for message in _errors(document))


def test_unknown_remix_mode_is_rejected() -> None:
    document = _job(
        "stem_remix",
        master_url="https://example.com/m.wav",
        stems=["https://example.com/d.wav"],
        remix_mode="widen_everything",
    )
    assert any("remix_mode" in message for message in _errors(document))


def test_unknown_master_profile_is_rejected() -> None:
    document = _job(
        "master_audio",
        audio_url="https://example.com/a.wav",
        profiles=["dynamic", "loud"],
    )
    assert any("unknown profiles" in message for message in _errors(document))


def test_naturalize_vocabulary_is_checked() -> None:
    document = _job(
        "naturalize",
        audio_url="https://example.com/a.wav",
        mode="aggressive",
        vocoder={"backend": "hifigan"},
    )
    messages = _errors(document)
    assert any("mode 'aggressive'" in message for message in messages)
    assert any("vocoder backend 'hifigan'" in message for message in messages)


def test_naturalize_out_of_range_intensity_warns() -> None:
    document = _job("naturalize", audio_url="https://example.com/a.wav", intensity=1.9)
    assert _errors(document) == []
    assert any("intensity 1.9" in message for message in _warnings(document))


def test_naturalize_percentage_intensity_is_accepted() -> None:
    document = _job("naturalize", audio_url="https://example.com/a.wav", intensity=110)
    assert not any("intensity" in message for message in _warnings(document))


def test_nested_full_pass_naturalize_block_is_checked() -> None:
    document = _job(
        "full_pass",
        master_url="https://example.com/m.wav",
        naturalize={"mode": "aggressive"},
    )
    assert any("mode 'aggressive'" in message for message in _errors(document))


def test_execution_timeout_above_ttl_is_rejected() -> None:
    document = _job("system_info")
    document["policy"] = {"ttl": 600000, "executionTimeout": 900000}
    assert any("exceeds" in message for message in _errors(document))


def test_non_integer_policy_is_rejected() -> None:
    document = _job("system_info")
    document["policy"] = {"ttl": "600000"}
    assert _errors(document) == ["policy.ttl must be an integer in milliseconds."]


def test_long_running_action_without_execution_timeout_warns() -> None:
    document = _job("stem_remix", master_url="https://x/m.wav", stems=["https://x/d.wav"])
    document["policy"] = {"ttl": 7200000}
    assert any("executionTimeout" in message for message in _warnings(document))


def test_committed_credentials_are_rejected() -> None:
    document = _job(
        "master_audio",
        audio_url="https://example.com/a.wav",
        github_release_export={
            "repository": "ledgercorp/stemforge-worker",
            "release_id": 1,
            "token": "ghp_0123456789abcdefghijklmnopqrstuvwxyz",
        },
    )
    messages = _errors(document)
    assert any("literal credential" in message for message in messages)


def test_secret_placeholders_are_allowed() -> None:
    document = _job(
        "master_audio",
        audio_url="https://example.com/a.wav",
        github_release_export={"token": "${{ secrets.GITHUB_TOKEN }}"},
    )
    assert _errors(document) == []


def test_base64_payloads_do_not_trip_the_secret_scan() -> None:
    document = _job(
        "naturalize",
        audio_base64="AKIA" + "A" * 16,
    )
    assert _errors(document) == []


def test_unknown_top_level_keys_warn() -> None:
    document = _job("system_info")
    document["inputs"] = {"action": "system_info"}
    assert any("unrecognized top-level keys" in message for message in _warnings(document))


def test_filename_convention_warns() -> None:
    assert any(
        "kebab-case" in message
        for message in _warnings(_job("system_info"), name="Quick Check.json")
    )
    assert not any(
        "kebab-case" in message
        for message in _warnings(_job("system_info"), name="live-check-20260804-1200.json")
    )


def test_cli_reports_errors_and_honours_strict(tmp_path: Path) -> None:
    good = tmp_path / "live-check-20260804-1200.json"
    good.write_text(json.dumps(_job("system_info")), encoding="utf-8")
    assert validator.main([str(good)]) == 0

    noisy = tmp_path / "Live Check.json"
    noisy.write_text(json.dumps(_job("system_info")), encoding="utf-8")
    assert validator.main([str(noisy)]) == 0
    assert validator.main([str(noisy), "--strict"]) == 1

    broken = tmp_path / "broken-20260804-1200.json"
    broken.write_text("{not json", encoding="utf-8")
    assert validator.main([str(broken)]) == 1


def test_cli_reports_missing_files(tmp_path: Path) -> None:
    assert validator.main([str(tmp_path / "absent-20260804-1200.json")]) == 1
