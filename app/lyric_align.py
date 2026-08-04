from __future__ import annotations

import base64
import binascii
import os
import re
from pathlib import Path

import torch
import whisperx
from rapidfuzz import fuzz

from app.export import build_lyric_exports
from app.memory import save_approved_timing
from app.utils import download_file, normalize_word, temporary_job_dir


ALIGNER_VERSION = "monotonic_dynamic_programming_v3"


def _flatten_words(segments: list[dict]) -> list[dict]:
    words: list[dict] = []
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
    tokens: list[dict] = []
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


def _word_similarity(expected_word: str, observed_word: str) -> float:
    if expected_word == observed_word:
        return 1.0
    return fuzz.ratio(expected_word, observed_word) / 100.0


def _anchor_threshold(token: str) -> float:
    if len(token) <= 2:
        return 0.999
    if len(token) == 3:
        return 0.82
    return 0.68


def _reconcile(expected: list[dict], observed: list[dict]) -> tuple[list[dict], dict]:
    """Align supplied lyric words to WhisperX words while preserving order.

    The earlier SequenceMatcher implementation could attach repeated fragments
    to later verses and could map several expected words to the same observed
    word. This global dynamic-programming alignment is strictly monotonic and
    one-to-one, with a mild positional penalty to disambiguate repeated lines.
    """
    expected_words = [item["normalized"] for item in expected]
    observed_words = [item["normalized"] for item in observed]
    n = len(expected_words)
    m = len(observed_words)

    if n == 0:
        return [], {
            "expected_word_count": 0,
            "detected_word_count": m,
            "recognized_or_fuzzy_anchor_count": 0,
            "anchor_coverage": 0.0,
            "sequence_similarity": 0.0,
            "alignment_method": ALIGNER_VERSION,
        }

    if m == 0:
        mapped = []
        for index, item in enumerate(expected):
            start = index * 0.4
            mapped.append(
                {
                    **item,
                    "start": start,
                    "end": start + 0.3,
                    "score": 0.0,
                    "source": "interpolated",
                }
            )
        return mapped, {
            "expected_word_count": n,
            "detected_word_count": 0,
            "recognized_or_fuzzy_anchor_count": 0,
            "anchor_coverage": 0.0,
            "sequence_similarity": 0.0,
            "alignment_method": ALIGNER_VERSION,
        }

    expected_gap_cost = 0.72
    observed_gap_cost = 0.48
    position_weight = 0.42

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + expected_gap_cost
        back[i][0] = "E"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + observed_gap_cost
        back[0][j] = "O"

    for i in range(1, n + 1):
        expected_fraction = (i - 1) / max(n - 1, 1)
        for j in range(1, m + 1):
            observed_fraction = (j - 1) / max(m - 1, 1)
            similarity = _word_similarity(
                expected_words[i - 1],
                observed_words[j - 1],
            )
            substitution_cost = (
                1.0
                - similarity
                + position_weight * abs(expected_fraction - observed_fraction)
            )

            options = (
                (dp[i - 1][j - 1] + substitution_cost, "M"),
                (dp[i - 1][j] + expected_gap_cost, "E"),
                (dp[i][j - 1] + observed_gap_cost, "O"),
            )
            best_cost, best_action = min(options, key=lambda item: item[0])
            dp[i][j] = best_cost
            back[i][j] = best_action

    pairs: list[tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        action = back[i][j]
        if action == "M":
            similarity = _word_similarity(
                expected_words[i - 1],
                observed_words[j - 1],
            )
            pairs.append((i - 1, j - 1, similarity))
            i -= 1
            j -= 1
        elif action == "E":
            i -= 1
        elif action == "O":
            j -= 1
        else:
            break

    pairs.reverse()
    mapped: list[dict | None] = [None] * n
    similarity_sum = 0.0
    similarity_count = 0

    for expected_index, observed_index, similarity in pairs:
        similarity_sum += similarity
        similarity_count += 1

        token = expected_words[expected_index]
        if similarity < _anchor_threshold(token):
            continue

        source = observed[observed_index]
        mapped[expected_index] = {
            **expected[expected_index],
            "start": source["start"],
            "end": source["end"],
            "score": min(float(source["score"]), similarity),
            "source": (
                "recognized_anchor" if similarity >= 0.999 else "fuzzy_anchor"
            ),
        }

    anchor_indices = [index for index, item in enumerate(mapped) if item is not None]

    for index, expected_item in enumerate(expected):
        if mapped[index] is not None:
            continue

        previous = max(
            (value for value in anchor_indices if value < index),
            default=None,
        )
        following = min(
            (value for value in anchor_indices if value > index),
            default=None,
        )

        if previous is not None and following is not None:
            left = mapped[previous]
            right = mapped[following]
            available = max(0.05, float(right["start"]) - float(left["end"]))
            step = available / max(following - previous, 1)
            start = float(left["end"]) + step * (index - previous - 0.15)
            duration = max(0.10, min(0.45, step * 0.72))
            end = min(float(right["start"]), start + duration)
        elif previous is not None:
            left = mapped[previous]
            start = float(left["end"]) + 0.06 + 0.34 * (index - previous - 1)
            end = start + 0.30
        elif following is not None:
            right = mapped[following]
            distance = following - index
            end = max(0.0, float(right["start"]) - 0.06 - 0.34 * (distance - 1))
            start = max(0.0, end - 0.30)
        else:
            start = index * 0.4
            end = start + 0.3

        mapped[index] = {
            **expected_item,
            "start": float(max(0.0, start)),
            "end": float(max(end, start + 0.05)),
            "score": 0.0,
            "source": "interpolated",
        }

    last_start = 0.0
    for item in mapped:
        item["start"] = max(float(item["start"]), last_start)
        item["end"] = max(float(item["end"]), item["start"] + 0.05)
        last_start = item["start"]

    recognized_count = sum(
        1
        for item in mapped
        if item["source"] in {"recognized_anchor", "fuzzy_anchor"}
    )
    coverage = recognized_count / max(n, 1)

    return mapped, {
        "expected_word_count": n,
        "detected_word_count": m,
        "recognized_or_fuzzy_anchor_count": recognized_count,
        "anchor_coverage": round(coverage, 4),
        "sequence_similarity": round(
            similarity_sum / max(similarity_count, 1),
            4,
        ),
        "alignment_method": ALIGNER_VERSION,
    }


def _build_lines(lyrics: str, mapped_words: list[dict]) -> list[dict]:
    text_lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    line_map: list[dict] = []

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
        start = float(min(item["start"] for item in words))
        end = float(max(item["end"] for item in words))
        span = end - start
        span_limit = max(6.0, len(words) * 1.8)
        suspicious_span = span > span_limit

        line_map.append(
            {
                "index": line_index + 1,
                "text": line_text,
                "start": start,
                "end": end,
                "confidence": round(float(confidence), 4),
                "anchor_coverage": round(len(anchored) / len(words), 4),
                "needs_review": (
                    len(anchored) / len(words) < 0.5
                    or confidence < 0.45
                    or suspicious_span
                ),
                "words": words,
            }
        )

    return line_map


def _write_embedded_audio(payload: dict, temp_path: Path) -> Path | None:
    encoded = payload.get("audio_base64")
    if not encoded:
        return None

    if not isinstance(encoded, str):
        raise ValueError("audio_base64 must be a base64 string.")

    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    encoded = "".join(encoded.split())
    if len(encoded) > 9_000_000:
        raise ValueError("Embedded audio exceeds the safe request-size limit.")

    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 is not valid base64.") from exc

    if not audio_bytes:
        raise ValueError("Embedded audio decoded to an empty file.")

    suffix = Path(str(payload.get("audio_filename") or "input_audio.opus")).suffix
    if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}:
        suffix = ".bin"
    audio_path = temp_path / f"input_audio{suffix}"
    audio_path.write_bytes(audio_bytes)
    return audio_path


def align_lyrics_job(payload: dict) -> dict:
    audio_url = payload.get("vocal_url") or payload.get("audio_url")
    audio_base64 = payload.get("audio_base64")
    lyrics = str(payload.get("lyrics") or "").strip()
    model_name = payload.get("model", "large-v3")
    language = payload.get("language", "en")
    artist = payload.get("artist", "sounddecay")
    song = payload.get("song", "untitled")
    approve = bool(payload.get("approve", False))

    if not audio_url and not audio_base64:
        return {
            "status": "rejected",
            "reason": "audio_url, vocal_url, or audio_base64 is required.",
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
        audio_path = _write_embedded_audio(payload, temp_path)

        if audio_path is None:
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
                "aligner_version": ALIGNER_VERSION,
                "metrics": metrics,
                "transcription_text": " ".join(word["word"] for word in observed),
            }

        lines = _build_lines(lyrics, mapped_words)
        exports = build_lyric_exports(
            artist=artist,
            song=song,
            lines=lines,
            output_dir=(
                Path(os.environ.get("STEMFORGE_WORKSPACE", "/tmp/stemforge"))
                / "output"
            ),
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
            "aligner_version": ALIGNER_VERSION,
            "metrics": metrics,
            "review_required": any(line["needs_review"] for line in lines),
            "review_line_indices": [
                line["index"] for line in lines if line["needs_review"]
            ],
            "lines": lines,
            "exports": exports,
        }
