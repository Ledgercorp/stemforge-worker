from __future__ import annotations

from app.analysis_v2 import analyze_audio_v2_job, compare_audio_v2_job
from app.audio_analysis import analyze_audio_url
from app.daw_v2 import export_daw_job
from app.lyric_align import ALIGNER_VERSION, align_lyrics_job
from app.lyrics_v2 import align_lyrics_smart_job
from app.mastering import mastering_plan as legacy_mastering_plan
from app.mastering_v2 import master_audio_job, mastering_plan
from app.memory import get_memory, record_feedback
from app.memory_v2 import get_production_profile_job, record_rule_job
from app.perceptual_listener import compare_audio_urls
from app.repair_v2 import humanize_audio_job, repair_audio_job
from app.separation_v2 import separate_stems_job
from app.stem_processing import inspect_stems_job
from app.stems_v2 import inspect_stems_v2_job
from app.storage import create_download_job, create_upload_job, delete_storage_objects, storage_status
from app.system_v2 import VERSION, system_info_job
from app.video_v2 import render_lyric_video_job
from app.volume_transfer import (
    volume_delete_job,
    volume_file_chunk_job,
    volume_file_info_job,
    volume_upload_chunk_job,
    volume_upload_finalize_job,
    volume_upload_init_job,
)

SUPPORTED_ACTIONS = {
    "system_info", "storage_info", "create_upload", "create_download", "delete_storage_objects",
    "volume_upload_init", "volume_upload_chunk", "volume_upload_finalize",
    "volume_file_info", "volume_file_chunk", "volume_delete",
    "align_lyrics", "align_lyrics_smart", "aligner_info",
    "analyze_audio", "analyze_audio_v2", "compare_audio", "compare_audio_v2",
    "inspect_stems", "inspect_stems_v2", "separate_stems",
    "mastering_plan", "legacy_mastering_plan", "master_audio", "repair_audio", "humanize_audio",
    "render_lyric_video", "export_daw",
    "get_memory", "record_feedback", "record_rule", "get_production_profile",
}


def handle_job(payload: dict) -> dict:
    action = str(payload.get("action", "system_info"))
    if action not in SUPPORTED_ACTIONS:
        return {
            "status": "rejected",
            "reason": f"Unsupported action: {action}",
            "stemforge_version": VERSION,
            "supported_actions": sorted(SUPPORTED_ACTIONS),
        }

    dispatch = {
        "system_info": lambda: system_info_job(payload),
        "storage_info": storage_status,
        "create_upload": lambda: create_upload_job(payload),
        "create_download": lambda: create_download_job(payload),
        "delete_storage_objects": lambda: delete_storage_objects(payload),
        "volume_upload_init": lambda: volume_upload_init_job(payload),
        "volume_upload_chunk": lambda: volume_upload_chunk_job(payload),
        "volume_upload_finalize": lambda: volume_upload_finalize_job(payload),
        "volume_file_info": lambda: volume_file_info_job(payload),
        "volume_file_chunk": lambda: volume_file_chunk_job(payload),
        "volume_delete": lambda: volume_delete_job(payload),
        "align_lyrics": lambda: align_lyrics_job(payload),
        "align_lyrics_smart": lambda: align_lyrics_smart_job(payload),
        "aligner_info": lambda: {
            "status": "completed",
            "aligner_version": ALIGNER_VERSION,
            "stemforge_version": VERSION,
        },
        "analyze_audio": lambda: analyze_audio_url(payload),
        "analyze_audio_v2": lambda: analyze_audio_v2_job(payload),
        "compare_audio": lambda: compare_audio_urls(payload),
        "compare_audio_v2": lambda: compare_audio_v2_job(payload),
        "inspect_stems": lambda: inspect_stems_job(payload),
        "inspect_stems_v2": lambda: inspect_stems_v2_job(payload),
        "separate_stems": lambda: separate_stems_job(payload),
        "mastering_plan": lambda: mastering_plan(payload),
        "legacy_mastering_plan": lambda: legacy_mastering_plan(payload),
        "master_audio": lambda: master_audio_job(payload),
        "repair_audio": lambda: repair_audio_job(payload),
        "humanize_audio": lambda: humanize_audio_job(payload),
        "render_lyric_video": lambda: render_lyric_video_job(payload),
        "export_daw": lambda: export_daw_job(payload),
        "get_memory": lambda: {
            "status": "completed",
            "memory": get_memory(payload.get("artist", "sounddecay")),
        },
        "record_feedback": lambda: record_feedback(payload),
        "record_rule": lambda: record_rule_job(payload),
        "get_production_profile": lambda: get_production_profile_job(payload),
    }
    result = dispatch[action]()
    if isinstance(result, dict):
        result.setdefault("stemforge_version", VERSION)
        result.setdefault("action", action)
    return result
