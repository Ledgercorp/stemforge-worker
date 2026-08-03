from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

import torch
import whisperx
from rapidfuzz import fuzz

from app.export import build_lyric_exports
from app.memory import save_approved_timing
from app.utils import download_file, normalize_word, temporary_job_dir


def _flatten_words(segments: list[dict]) -> list[dict]:
    words = []
    for segment in segments:
        for word in segment.get("words") or []:
            text = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")
            if not text or start is None or end is None:
                continue

            normalized = normalize_word(text)
            if not normalized:
                continue

            words.append(
                {
                    "word": text,
                    "normalized": normalized,
                    "start": float(start),
                    "end": float(end),
                    "score": float(word.get("score") or 0.0),
                    "source": "whisperx",
                }
            )
    return words


def _expected_tokens(lyrics: str) -> list[dict]:
    tokens = []
    line_index = 0

    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for match in re.finditer(r"[A-Za-z0-9']+", line):
            normalized = normalize_word(match.group(0))
            if normalized:
                tokens.append(
                    {
                        "word": match.group(0),
                        "normalized": normalized,
                        "line_index": line_index,
                    }
                )

        line_index += 1

    return tokens


def _reconcile(expected: list[dict], observed: list[dict]) -> tuple[list[dict], dict]:
    expected_words = [item["normalized"] for item in expected]
    observed_words = [item["normalized"] for item in observed]

    matcher = difflib.SequenceMatcher(
        a=expected_words,
        b=observed_words,
        autojunk=False,
    )

    mapped: list[dict | None] = [None] * len(expected)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                source = observed[j1 + offset]
                mapped[i1 + offset] = {
                    **expected[i1 + offset],
                    "start": source["start"],
                    "end": source["end"],
                    "score": source["score"],
                    "source": "recognized_anchor",
                }
            continue

        if tag == "replace":
            expected_span = expected[i1:i2]
            observed_span = observed[j1:j2]
            for index, expected_item in enumerate(expected_span):
                best = None
                best_score = 0.0
                for observed_item in observed_span:
                    similarity = fuzz.ratio(
                        expected_item["normalized"],
                        observed_item["normalized"],
                    )
                    if similarity > best_score:
                        best = observed_item
                        best_score = similarity

                if best is not None and best_score >= 72:
                    mapped[i1 + index] = {
                        **expected_item,
                        "start": best["start"],
                        "end": best["end"],
                        "score": min(best["score"], best_score / 100),
                        "source": "fuzzy_anchor",
                    }

    anchor_indices = [index for index, item in enumerate(mapped) if item is not None]

    for index, expected_item in enumerate(expected):
        if mapped[index] is not None:
            continue

        previous = max((value for value in anchor_indices if value < index), default=None)
        following = min((value for value in anchor_indices if value > index), default=None)

        if previous is not None and following is not None:
            left = mapped[previous]
            right = mapped[following]
            fraction = (index - previous) / (following - previous)
            start = left["end"] + (right["start"] - left["end"]) * fraction
            duration = max(
                0.12,
                (right["start"] - left["end"])
                / max(following - previous, 1)
                * 0.8,
            )
            end = min(right["start"], start + duration)
        elif previous is not None:
            left = mapped[previous]
            start = left["end"] + 0.06
            end = start + 0.35
        elif following is not None:
            right = mapped[following]
            end = max(0.0, right["start"] - 0.06)
            start = max(0.0, end - 0.35)
        else:
            start = index * 0.4
            end = start + 0.3

        mapped[index] = {
            **expected_item,
            "start": float(start),
            "end": float(max(end, start + 0.05)),
            "score": 0.0,
            "source": "interpolated",
        }

    recognized_count = sum(
        1 for item in mapped if item["source"] in {"recognized_anchor", "fuzzy_anchor"}
    )
    coverage = recognized_count / max(len(expected), 1)

    return mapped, {
        "expected_word_count": len(expected),
        "detected_word_count": len(observed),
        "recognized_or_fuzzy_anchor_count": recognized_count,
        "anchor_coverage": round(coverage, 4),
        "sequence_similarity": round(matcher.ratio(), 4),
    }


def _build_lines(lyrics: str, mapped_words: list[dict]) -> list[dict]:
    text_lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    line_map = []

    for line_index, line_text in enumerate(text_lines):
        words = [item for item in mapped_words if item["line_index"] == line_index]
        if not words:
            continue

        anchored = [item for item in words if item["source"] != "interpolated"]
        confidence = (
            sum(item["score"] for item in anchored) / len(anchored)
            if anchored
            else 0.0
        )

        line_map.append(
            {
                "index": line_index + 1,
                "text": line_text,
                "start": float(min(item["start"] for item in words)),
                "end": float(max(item["end"] for item in words)),
                "confidence": round(float(confidence), 4),
                "anchor_coverage": round(len(anchored) / len(words), 4),
                "needs_review": len(anchored) / len(words) < 0.5 or confidence < 0.45,
                "words": words,
            }
        )

    return line_map


def align_lyrics_job(payload: dict) -> dict:
    audio_url = payload.get("vocal_url") or payload.get("audio_url")
    lyrics = str(payload.get("lyrics") or "").strip()
    model_name = payload.get("model", "large-v3")
    language = payload.get("language", "en")
    artist = payload.get("artist", "sounddecay")
    song = payload.get("song", "untitled")
    approve = bool(payload.get("approve", False))

    if not audio_url:
        return {
            "status": "rejected",
            "reason": "audio_url or vocal_url is required.",
        }

    if not lyrics:
        return {
            "status": "rejected",
            "reason": "lyrics is required.",
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = int(payload.get("batch_size", 8 if device == "cuda" else 2))

    with temporary_job_dir("lyrics") as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / "input_audio"
        download_file(audio_url, audio_path)

        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=language,
        )
        audio = whisperx.load_audio(str(audio_path))
        transcription = model.transcribe(audio, batch_size=batch_size)

        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

        aligned = whisperx.align(
            transcription["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=True,
        )

        observed = _flatten_words(aligned["segments"])
        expected = _expected_tokens(lyrics)
        mapped_words, metrics = _reconcile(expected, observed)

        if metrics["anchor_coverage"] < float(
            payload.get("minimum_anchor_coverage", 0.35)
        ):
            return {
                "status": "rejected",
                "reason": (
                    "Too few lyric words were confidently aligned. "
                    "Confirm the lyrics and audio are the same version."
                ),
                "device": device,
                "model": model_name,
                "metrics": metrics,
                "transcription_text": " ".join(word["word"] for word in observed),
            }

        lines = _build_lines(lyrics, mapped_words)
        exports = build_lyric_exports(
            artist=artist,
            song=song,
            lines=lines,
            output_dir=Path(
                os.environ.get("STEMFORGE_WORKSPACE", "/tmp/stemforge")
            )
            / "output",
        )

        if approve:
            save_approved_timing(
                artist=artist,
                song=song,
                timing={
                    "metrics": metrics,
                    "lines": lines,
                    "exports": exports,
                },
            )

        return {
            "status": "completed",
            "device": device,
            "model": model_name,
            "language": language,
            "metrics": metrics,
            "review_required": any(line["needs_review"] for line in lines),
            "review_line_indices": [
                line["index"] for line in lines if line["needs_review"]
            ],
            "lines": lines,
            "exports": exports,
        }
