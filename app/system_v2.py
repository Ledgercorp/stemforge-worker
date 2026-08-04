from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

from app import numpy_json_patch as _numpy_json_patch
from app.naturalize_dsp_v2 import (
    NATURALIZE_PRESET,
    NOMINAL_COCKTAIL,
    PARAMETER_RANGES,
    ROLE_MULTIPLIERS,
    processing_order,
)
from app.storage import storage_status
from app.vocoder_v2 import availability

VERSION = "2.4.0"
BUILD = "v2.4.0-naturalize-balanced-vocoder"


def _package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def system_info_job(payload: dict | None = None) -> dict:
    del payload
    packages = {
        name: _package_available(name)
        for name in [
            "whisperx",
            "torch",
            "demucs",
            "pyloudnorm",
            "boto3",
            "mido",
            "scipy",
            "vocos",
            "bigvgan",
        ]
    }
    executables = {
        name: shutil.which(name) is not None
        for name in ["ffmpeg", "ffprobe"]
    }
    vocoder_backends = {
        backend: availability(backend)
        for backend in ("vocos", "bigvgan", "discoder")
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
            "naturalize_warble_risk_detection": True,
            "naturalize_pumping_detection": True,
            "naturalize_metallic_artifact_detection": True,
            "naturalize_clarity_loss_detection": True,
            "naturalize_transient_retention_detection": True,
            "vocoder_aliasing_and_transient_gate": True,
            "automatic_intensity_reduction": True,
            "vocoder_fallback_to_dsp": True,
        },
        "reliability_guards": {
            "atomic_input_downloads": True,
            "input_retry_and_stale_link_detection": True,
            "optional_size_and_sha256_validation": True,
            "drive_head_to_range_get_preflight_fallback": True,
            "single_canonical_stem_remix_workflow": True,
            "dynamic_workflow_asset_counts": True,
            "failure_diagnostic_artifacts": True,
            "lazy_optional_vocoder_imports": True,
        },
        "naturalize": {
            "label": "Naturalness / Authenticity enhancement",
            "preset": NATURALIZE_PRESET,
            "callable_action": "naturalize",
            "modes": ["auto", "quick", "surgical"],
            "default_mode": "auto",
            "default_intensity": 1.0,
            "intensity_range": [0.6, 1.4],
            "explicit_zero_bypass": True,
            "maximum_full_passes": 2,
            "second_pass_maximum_intensity_scale": 0.70,
            "non_destructive_default": True,
            "ab_reference_default": True,
            "pre_denoise_reference_default": True,
            "parameter_ranges": PARAMETER_RANGES,
            "nominal_cocktail": NOMINAL_COCKTAIL,
            "stem_depth_multipliers": ROLE_MULTIPLIERS,
            "nominal_processing_order": processing_order(),
            "denoise_constraints": {
                "type": "adaptive_music_spectral",
                "position": "post_cocktail_only",
                "maximum_reduction_db_range": [3.0, 8.0],
                "default_reduction_db": 5.0,
                "quiet_profile_ms_range": [200.0, 500.0],
                "default_quiet_profile_ms": 350.0,
                "transient_protection": True,
                "dry_microvariation_preservation_mix": 0.10,
                "spectral_uniformity_restoration": False,
            },
            "additional_layers": {
                "shaped_noise_floor_dbfs_range": [-72.0, -60.0],
                "noise_shapes": ["pink", "tape_hiss"],
                "micro_timing_jitter_ms_range": [1.5, 3.0],
                "velocity_sensitive_timing": True,
                "low_drive_gain_compensated_saturation": True,
                "frequency_dependent_midrange_depth": True,
                "optional_gentle_multiband_compression": True,
            },
            "vocoder": {
                "default_policy": "vocos_on_vocals_in_surgical_when_available",
                "position": "after_cocktail_and_constrained_denoise",
                "backends": {
                    "vocos": {
                        "model": "charactr/vocos-mel-24khz",
                        "path": "default_fast",
                        **vocoder_backends["vocos"],
                    },
                    "bigvgan": {
                        "model": "nvidia/bigvgan_v2_44khz_128band_512x",
                        "path": "explicit_high_quality",
                        **vocoder_backends["bigvgan"],
                    },
                    "discoder": {
                        "model": "disco-eth/discoder",
                        "path": "explicit_max_music_fidelity",
                        "environment_configured": all(
                            os.environ.get(name)
                            for name in (
                                "STEMFORGE_DISCODER_ROOT",
                                "STEMFORGE_DISCODER_CHECKPOINT",
                                "STEMFORGE_DISCODER_CONFIG",
                            )
                        ),
                        **vocoder_backends["discoder"],
                    },
                },
                "default_wet_by_role": {
                    "vocal": 0.85,
                    "harmonic": 0.60,
                    "percussion": 0.20,
                    "other": 0.45,
                },
                "quality_gate_fallback_to_dsp": True,
                "pre_vocoder_intermediate_retained_in_report": True,
            },
            "agent_policy": {
                "prefer_surgical_when_stems_are_available": True,
                "fall_back_to_quick_only_when_necessary": True,
                "surgical_can_run_demucs_when_explicitly_requested": True,
                "denoise_must_follow_all_modulation": True,
                "vocoder_must_follow_denoise": True,
                "automatically_reduce_intensity_or_abort": True,
                "classification": "fidelity_authenticity_enhancement",
                "not_a_detection_tool": True,
                "not_a_detection_evasion_tool": True,
                "musical_transparency_has_priority": True,
            },
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
                "naturalize",
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
