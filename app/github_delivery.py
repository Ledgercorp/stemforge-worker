from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import quote

import requests

from app.mastering_v2 import master_audio_job


def _upload_release_asset(
    *,
    repository: str,
    release_id: int,
    token: str,
    path: Path,
    asset_name: str,
) -> dict:
    if "/" not in repository:
        raise ValueError("github_repository must be in owner/repo form.")
    safe_name = Path(asset_name).name
    url = (
        f"https://uploads.github.com/repos/{repository}/releases/"
        f"{int(release_id)}/assets?name={quote(safe_name)}"
    )
    content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    size = path.stat().st_size
    with path.open("rb") as source:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": content_type,
                "Content-Length": str(size),
            },
            data=source,
            timeout=(30, 3600),
        )
    if response.status_code != 201:
        detail = response.text[:1000]
        raise RuntimeError(
            f"GitHub release asset upload failed for {safe_name}: "
            f"HTTP {response.status_code}: {detail}"
        )
    item = response.json()
    return {
        "asset_id": item.get("id"),
        "name": item.get("name"),
        "size_bytes": item.get("size"),
        "state": item.get("state"),
        "browser_download_url": item.get("browser_download_url"),
    }


def master_audio_github_delivery_job(payload: dict) -> dict:
    token = str(payload.get("github_token") or "").strip()
    repository = str(payload.get("github_repository") or "").strip()
    release_id = int(payload.get("github_release_id") or 0)
    if not token:
        return {"status": "rejected", "reason": "github_token is required."}
    if "/" not in repository:
        return {
            "status": "rejected",
            "reason": "github_repository must be in owner/repo form.",
        }
    if release_id <= 0:
        return {"status": "rejected", "reason": "github_release_id is required."}

    clean_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"github_token", "github_repository", "github_release_id"}
    }
    result = master_audio_job(clean_payload)
    if result.get("status") != "completed":
        return result

    prefix = str(payload.get("asset_prefix") or "StemForge").strip() or "StemForge"
    assets: list[dict] = []
    for candidate in result.get("candidates") or []:
        profile = str(candidate.get("profile") or "candidate").title()
        local_path = Path(str(candidate.get("local_path") or ""))
        if not local_path.is_file():
            raise FileNotFoundError(str(local_path))
        asset = _upload_release_asset(
            repository=repository,
            release_id=release_id,
            token=token,
            path=local_path,
            asset_name=f"{prefix}_{profile}.wav",
        )
        asset["profile"] = str(candidate.get("profile") or "")
        assets.append(asset)

    report_path = Path(str((result.get("report_output") or {}).get("local_path") or ""))
    if not report_path.is_file():
        report_path = Path("/tmp/stemforge_mastering_v2/mastering_report.json")
    if report_path.is_file():
        assets.append(
            _upload_release_asset(
                repository=repository,
                release_id=release_id,
                token=token,
                path=report_path,
                asset_name=f"{prefix}_Mastering_Report.json",
            )
        )

    compact_candidates = []
    for candidate in result.get("candidates") or []:
        compact_candidates.append(
            {
                "profile": candidate.get("profile"),
                "chain": candidate.get("chain"),
                "metrics": {
                    key: (candidate.get("metrics") or {}).get(key)
                    for key in (
                        "duration_seconds",
                        "sample_rate",
                        "channels",
                        "integrated_lufs",
                        "true_peak_dbtp",
                        "sample_peak_dbfs",
                        "rms_dbfs",
                        "crest_factor_db",
                        "stereo_correlation",
                        "clipped_sample_count",
                        "waveform_discontinuity_count",
                    )
                },
                "safety": candidate.get("safety"),
            }
        )

    return {
        "status": "completed",
        "engine": "StemForge v2 GitHub release delivery",
        "artist": result.get("artist"),
        "song": result.get("song"),
        "source_metrics": {
            key: (result.get("source_metrics") or {}).get(key)
            for key in (
                "duration_seconds",
                "sample_rate",
                "channels",
                "integrated_lufs",
                "true_peak_dbtp",
                "sample_peak_dbfs",
                "rms_dbfs",
                "crest_factor_db",
                "stereo_correlation",
                "clipped_sample_count",
            )
        },
        "candidates": compact_candidates,
        "github_release_id": release_id,
        "github_repository": repository,
        "assets": assets,
        "delivery_complete": True,
    }
