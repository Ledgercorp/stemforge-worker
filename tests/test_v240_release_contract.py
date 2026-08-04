from pathlib import Path


def test_v240_release_and_runpod_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    relay = (root / ".github/workflows/run-stemforge-stem-remix.yml").read_text(
        encoding="utf-8"
    )
    release = (root / ".github/workflows/publish-v2-4-0.yml").read_text(
        encoding="utf-8"
    )
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()

    assert 'minimum_worker_version // "2.4.0"' in relay
    assert (
        'required_build // "v2.4.0-naturalize-balanced-vocoder"'
        in relay
    )
    assert "Publish StemForge v2.4.0" in release
    assert "v2.4.0" in release
    assert "vocos" in requirements
    assert "huggingface_hub" in requirements
