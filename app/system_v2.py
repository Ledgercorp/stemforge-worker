from __future__ import annotations

import importlib.util
import shutil

from app import numpy_json_patch as _numpy_json_patch
from app.storage import storage_status

VERSION = "2.2.1"
BUILD = "v2.2.1-quality-hotfix"


def system_info_job(payload: dict | None = None) -> dict:
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in [
            "whisperx",
            "torch",
            "demucs",
            "pyloudnorm",
            "boto3",
            "mido",
            "scipy",
        ]
    }
    executables = {
        name: shutil.which(name) is not None
        for name in ["ffmpeg", "ffprobe"]
    }
    return {
        "status": "completed",
        "stemforge_version": VERSION,
        "build": BUILD,
        "packages": packages,
        "executables": executables,
        "storage": storage_status(),
        "quality_gates": {
            "artifact_discontinuity_detection": True,
            "high_frequency_noise_growth_detection": True,
            "level_matched_residual_comparison": True,
            "strict_stem_reconstruction_gate": True,
            "master_anchored_delta_fallback": True,
            "non_finite_sample_rejection": True,
            "strict_output_write_available": True,
        },
        "reliability_guards": {
            "atomic_input_downloads": True,
            "input_retry_and_stale_link_detection": True,
            "optional_size_and_sha256_validation": True,
            "dynamic_workflow_asset_counts": True,
            "failure_diagnostic_artifacts": True,
        },
        "feature_groups": {
            "lyrics": ["align_lyrics", "align_lyrics_smart"],
            "analysis": [
                "analyze_audio",
                "analyze_audio_v2",
                "compare_audio",
                "compare_audio_v2",
            ],
            "stems": [
                "inspect_stems",
                "inspect_stems_v2",
                "separate_stems",
                "stem_remix",
            ],
            "processing": [
                "master_audio",
                "master_audio_github_delivery",
                "full_pass",
                "stem_remix",
                "repair_audio",
                "humanize_audio",
            ],
            "visual": ["render_lyric_video"],
            "interchange": ["export_daw"],
            "transfer": [
                "create_upload",
                "create_download",
                "delete_storage_objects",
                "volume_upload_init",
                "volume_upload_chunk",
                "volume_upload_finalize",
                "volume_file_info",
                "volume_file_chunk",
                "volume_delete",
                "master_audio_github_delivery",
                "private_github_release_export",
            ],
            "memory": [
                "get_memory",
                "record_feedback",
                "record_rule",
                "get_production_profile",
            ],
        },
    }
