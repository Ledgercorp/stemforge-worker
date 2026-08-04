from __future__ import annotations

import importlib.util
import shutil

from app.storage import storage_status

VERSION = "2.1.0"


def system_info_job(payload: dict | None = None) -> dict:
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ["whisperx", "torch", "demucs", "pyloudnorm", "boto3", "mido"]
    }
    executables = {
        name: shutil.which(name) is not None
        for name in ["ffmpeg", "ffprobe"]
    }
    return {
        "status": "completed",
        "stemforge_version": VERSION,
        "packages": packages,
        "executables": executables,
        "storage": storage_status(),
        "feature_groups": {
            "lyrics": ["align_lyrics", "align_lyrics_smart"],
            "analysis": ["analyze_audio", "analyze_audio_v2", "compare_audio", "compare_audio_v2"],
            "stems": ["inspect_stems", "inspect_stems_v2", "separate_stems"],
            "processing": ["master_audio", "master_audio_github_delivery", "full_pass", "repair_audio", "humanize_audio"],
            "visual": ["render_lyric_video"],
            "interchange": ["export_daw"],
            "transfer": [
                "create_upload", "create_download", "delete_storage_objects",
                "volume_upload_init", "volume_upload_chunk", "volume_upload_finalize",
                "volume_file_info", "volume_file_chunk", "volume_delete",
                "master_audio_github_delivery", "private_github_release_export",
            ],
            "memory": ["get_memory", "record_feedback", "record_rule", "get_production_profile"],
        },
    }
