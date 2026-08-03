from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return value or "untitled"


def _srt_time(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def _ass_time(seconds: float) -> str:
    centiseconds = int(round(max(0.0, seconds) * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"


def _lrc_time(seconds: float) -> str:
    centiseconds = int(round(max(0.0, seconds) * 100))
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"[{minutes:02}:{secs:02}.{centiseconds:02}]"


def build_lyric_exports(
    artist: str,
    song: str,
    lines: list[dict],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(artist)}_{_slug(song)}"

    srt_path = output_dir / f"{stem}.srt"
    ass_path = output_dir / f"{stem}.ass"
    lrc_path = output_dir / f"{stem}.lrc"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    srt_blocks = []
    for line in lines:
        srt_blocks.extend(
            [
                str(line["index"]),
                f"{_srt_time(line['start'])} --> {_srt_time(line['end'])}",
                line["text"],
                "",
            ]
        )
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    ass_header = """[Script Info]
Title: StemForge Lyrics
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,Arial,62,&H00FFFFFF,&H00FFFFFF,&H80000000,&H90000000,0,0,0,0,100,100,2,0,1,2,0,2,120,120,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_events = []
    for line in lines:
        karaoke = []
        for word in line["words"]:
            duration_cs = max(1, int(round((word["end"] - word["start"]) * 100)))
            karaoke.append(f"{{\\k{duration_cs}}}{word['word']} ")
        ass_events.append(
            f"Dialogue: 0,{_ass_time(line['start'])},{_ass_time(line['end'])},"
            f"Lyrics,,0,0,0,,{''.join(karaoke).strip()}"
        )
    ass_path.write_text(ass_header + "\n".join(ass_events), encoding="utf-8")

    lrc = [f"[ti:{song}]", f"[ar:{artist}]", "[re:StemForge WhisperX]"]
    lrc.extend(f"{_lrc_time(line['start'])}{line['text']}" for line in lines)
    lrc_path.write_text("\n".join(lrc), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "index",
                "start_seconds",
                "end_seconds",
                "confidence",
                "anchor_coverage",
                "needs_review",
                "lyric",
            ]
        )
        for line in lines:
            writer.writerow(
                [
                    line["index"],
                    f"{line['start']:.3f}",
                    f"{line['end']:.3f}",
                    f"{line['confidence']:.4f}",
                    f"{line['anchor_coverage']:.4f}",
                    line["needs_review"],
                    line["text"],
                ]
            )

    json_path.write_text(
        json.dumps({"artist": artist, "song": song, "lines": lines}, indent=2),
        encoding="utf-8",
    )

    return {
        "srt_path": str(srt_path),
        "ass_path": str(ass_path),
        "lrc_path": str(lrc_path),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
