from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy import signal

from app.naturalize_dsp_v2 import align_audio, artifact_assessment, clamp, match_level

VOCODER_DEFAULTS = {
    "backend": "vocos",
    "model": "charactr/vocos-mel-24khz",
    "feature_perturbation": 0.0025,
    "wet_by_role": {
        "vocal": 0.85,
        "harmonic": 0.60,
        "percussion": 0.20,
        "other": 0.45,
        "full_mix": 0.35,
    },
}

_MODELS: dict[tuple[str, str, str], Any] = {}


def resolve_vocoder(payload: dict, mode: str) -> dict:
    raw = payload.get("vocoder", payload.get("vocoder_flag", "auto"))
    if raw in (False, None, "off", "none", "disabled"):
        return {"enabled": False, "requested": raw, "backend": None}
    if raw is True:
        raw = "auto"
    config = dict(raw) if isinstance(raw, dict) else {"backend": str(raw).lower()}
    backend = str(config.get("backend") or "auto").lower()
    if backend == "auto":
        backend = "vocos" if mode == "surgical" else "off"
    if backend == "off":
        return {"enabled": False, "requested": raw, "backend": None}
    if backend not in {"vocos", "bigvgan", "discoder"}:
        raise ValueError("vocoder must be auto, off, vocos, bigvgan, or discoder.")
    models = {
        "vocos": "charactr/vocos-mel-24khz",
        "bigvgan": "nvidia/bigvgan_v2_44khz_128band_512x",
        "discoder": "disco-eth/discoder",
    }
    custom_wet = config.get("wet_by_role") or {}
    return {
        "enabled": True,
        "requested": raw,
        "backend": backend,
        "model": str(config.get("model") or models[backend]),
        "required": bool(config.get("required", False)),
        "compute_budget": str(config.get("compute_budget", payload.get("compute_budget", "auto"))).lower(),
        "feature_perturbation": clamp(
            config.get("feature_perturbation", payload.get("vocoder_feature_perturbation", 0.0025)),
            0.0,
            0.01,
            0.0025,
        ),
        "wet_by_role": {
            role: clamp(custom_wet.get(role, value), 0, 1, value)
            for role, value in VOCODER_DEFAULTS["wet_by_role"].items()
        },
    }


def availability(backend: str) -> dict:
    if backend == "vocos":
        modules = ["torch", "vocos"]
        return {"available": all(importlib.util.find_spec(x) is not None for x in modules), "required_modules": modules}
    if backend == "bigvgan":
        modules = ["torch", "bigvgan", "meldataset"]
        return {"available": all(importlib.util.find_spec(x) is not None for x in modules), "required_modules": modules}
    variables = ["STEMFORGE_DISCODER_ROOT", "STEMFORGE_DISCODER_CHECKPOINT", "STEMFORGE_DISCODER_CONFIG"]
    root, checkpoint, config = (os.environ.get(x) for x in variables)
    ready = bool(root and checkpoint and config and Path(root, "inference.py").exists() and Path(checkpoint).exists() and Path(config).exists())
    return {"available": ready, "required_environment": variables}


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return signal.resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def _align_mono(audio: np.ndarray, length: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(mono) == length:
        return mono
    out = np.zeros(length, np.float32)
    out[: min(length, len(mono))] = mono[: min(length, len(mono))]
    return out


def _chunked(audio: np.ndarray, sr: int, transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    chunk = max(1, int(sr * 12))
    overlap = min(int(sr * 0.25), chunk // 4)
    if len(audio) <= chunk:
        return _align_mono(transform(audio), len(audio))
    step = chunk - overlap
    out = np.zeros(len(audio), np.float64)
    weight = np.zeros(len(audio), np.float64)
    for start in range(0, len(audio), step):
        stop = min(len(audio), start + chunk)
        generated = _align_mono(transform(audio[start:stop]), stop - start)
        window = np.ones(stop - start)
        if overlap:
            fade = np.linspace(0, 1, overlap, endpoint=False)
            if start:
                window[:overlap] = fade
            if stop < len(audio):
                window[-overlap:] = fade[::-1]
        out[start:stop] += generated * window
        weight[start:stop] += window
        if stop == len(audio):
            break
    return (out / np.maximum(weight, 1e-12)).astype(np.float32)


def _run_vocos(mono: np.ndarray, sr: int, model_id: str, perturbation: float, seed: int) -> tuple[np.ndarray, dict]:
    import torch
    from vocos import Vocos

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ("vocos", model_id, device)
    model = _MODELS.get(key)
    if model is None:
        model = Vocos.from_pretrained(model_id).eval().to(device)
        _MODELS[key] = model
    target_rate = 24000
    source = _resample(mono, sr, target_rate)
    rng = np.random.default_rng(seed)

    def transform(chunk: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            if perturbation > 0:
                features = model.feature_extractor(tensor)
                noise = torch.from_numpy(rng.standard_normal(tuple(features.shape)).astype(np.float32)).to(device)
                generated = model.decode(features * (1 + perturbation * torch.tanh(noise)))
            else:
                generated = model(tensor)
        return generated.squeeze().detach().cpu().float().numpy()

    generated = _resample(_chunked(source, target_rate, transform), target_rate, sr)
    return _align_mono(generated, len(mono)), {
        "backend": "vocos",
        "model": model_id,
        "device": device,
        "model_sample_rate": target_rate,
        "feature_perturbation": round(perturbation, 7),
        "chunk_seconds": 12.0,
        "overlap_seconds": 0.25,
    }


def _run_bigvgan(mono: np.ndarray, sr: int, model_id: str, perturbation: float, seed: int) -> tuple[np.ndarray, dict]:
    import torch
    import bigvgan
    from meldataset import get_mel_spectrogram

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ("bigvgan", model_id, device)
    model = _MODELS.get(key)
    if model is None:
        model = bigvgan.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False)
        model.remove_weight_norm()
        model = model.eval().to(device)
        _MODELS[key] = model
    target_rate = int(model.h.sampling_rate)
    source = _resample(mono, sr, target_rate)
    rng = np.random.default_rng(seed)

    def transform(chunk: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            mel = get_mel_spectrogram(tensor, model.h)
            if perturbation > 0:
                noise = torch.from_numpy(rng.standard_normal(tuple(mel.shape)).astype(np.float32)).to(device)
                mel = mel * (1 + perturbation * torch.tanh(noise))
            generated = model(mel)
        return generated.squeeze().detach().cpu().float().numpy()

    generated = _resample(_chunked(source, target_rate, transform), target_rate, sr)
    return _align_mono(generated, len(mono)), {
        "backend": "bigvgan",
        "model": model_id,
        "device": device,
        "model_sample_rate": target_rate,
        "feature_perturbation": round(perturbation, 7),
        "chunk_seconds": 12.0,
        "overlap_seconds": 0.25,
    }


def _run_discoder(mono: np.ndarray, sr: int, model_id: str, perturbation: float, seed: int) -> tuple[np.ndarray, dict]:
    del model_id, perturbation, seed
    root = Path(os.environ["STEMFORGE_DISCODER_ROOT"])
    checkpoint = Path(os.environ["STEMFORGE_DISCODER_CHECKPOINT"])
    config = Path(os.environ["STEMFORGE_DISCODER_CONFIG"])
    with tempfile.TemporaryDirectory(prefix="stemforge_discoder_") as temp:
        temp = Path(temp)
        input_dir, output_dir = temp / "input", temp / "output"
        input_dir.mkdir(); output_dir.mkdir()
        source = _resample(mono, sr, 44100)
        sf.write(input_dir / "input.wav", source, 44100, subtype="PCM_24")
        process = subprocess.run(
            [
                "python", "-u", str(root / "inference.py"),
                "--input_dir", str(input_dir),
                "--output_dir", str(output_dir),
                "--checkpoint_file", str(checkpoint),
                "--config", str(config),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=7200,
        )
        rendered_path = next(iter(output_dir.rglob("*.wav")), None)
        if rendered_path is None:
            raise RuntimeError("DisCoder completed without a WAV output.")
        rendered, rendered_rate = sf.read(rendered_path, dtype="float32", always_2d=False)
        if np.asarray(rendered).ndim > 1:
            rendered = np.mean(rendered, axis=1)
        rendered = _resample(np.asarray(rendered, np.float32), int(rendered_rate), sr)
        return _align_mono(rendered, len(mono)), {
            "backend": "discoder",
            "model": "disco-eth/discoder",
            "model_sample_rate": 44100,
            "feature_perturbation": 0.0,
            "stdout_tail": process.stdout[-1000:],
        }


RUNNERS: dict[str, Callable[..., tuple[np.ndarray, dict]]] = {
    "vocos": _run_vocos,
    "bigvgan": _run_bigvgan,
    "discoder": _run_discoder,
}


def apply_vocoder(audio: np.ndarray, sr: int, config: dict, role: str, intensity: float, seed: int) -> tuple[np.ndarray, dict]:
    backend = config.get("backend")
    if not config.get("enabled") or not backend:
        return audio.copy(), {"enabled": False, "backend": None, "bypassed": True, "reason": "disabled", "wet_mix": 0.0}
    ready = availability(str(backend))
    if not ready["available"]:
        if config.get("required"):
            raise RuntimeError(f"Requested vocoder backend '{backend}' is unavailable.")
        return audio.copy(), {"enabled": True, "backend": backend, "model": config["model"], "bypassed": True, "reason": "backend_unavailable", "availability": ready, "wet_mix": 0.0}
    wet = float(np.clip(config["wet_by_role"].get(role, config["wet_by_role"]["other"]) * intensity, 0, 1))
    if wet <= 0:
        return audio.copy(), {"enabled": True, "backend": backend, "model": config["model"], "bypassed": True, "reason": "zero_wet_mix", "wet_mix": 0.0}
    mid = np.mean(audio, axis=1, dtype=np.float32)
    side = (audio[:, 0] - audio[:, 1]) * 0.5
    generated, model_report = RUNNERS[str(backend)](
        mid,
        sr,
        config["model"],
        config["feature_perturbation"] * intensity,
        seed,
    )
    reconstructed = np.column_stack((generated + side, generated - side)).astype(np.float32)
    candidate, level = match_level(audio, audio * (1 - wet) + reconstructed * wet, 0.75)
    safety = artifact_assessment(audio, candidate, sr, intensity, stage="vocoder")
    if not safety["accepted"]:
        return audio.copy(), {
            "enabled": True,
            "backend": backend,
            "model": config["model"],
            "bypassed": True,
            "reason": "quality_gate_fallback_to_dsp",
            "wet_mix": round(wet, 4),
            "model_report": model_report,
            "safety": safety,
            "level": level,
        }
    return align_audio(candidate, len(audio)), {
        "enabled": True,
        "backend": backend,
        "model": config["model"],
        "bypassed": False,
        "wet_mix": round(wet, 4),
        "feature_perturbation": round(config["feature_perturbation"] * intensity, 7),
        "model_report": model_report,
        "safety": safety,
        "level": level,
    }
