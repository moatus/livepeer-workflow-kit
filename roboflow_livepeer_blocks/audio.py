"""Local audio file chunking helpers."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class AudioChunk:
    index: int
    path: Path
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    temporary: bool


def probe_audio_duration_seconds(audio_path: str | Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Audio duration must be positive for {audio_path}")
    return duration


def plan_audio_chunks(
    audio_path: str | Path,
    chunk_size_seconds: float = 10.0,
    duration_seconds: Optional[float] = None,
) -> List[AudioChunk]:
    if chunk_size_seconds <= 0:
        raise ValueError("chunk_size_seconds must be positive")
    source_path = Path(audio_path)
    duration = (
        probe_audio_duration_seconds(source_path)
        if duration_seconds is None
        else duration_seconds
    )
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    chunk_count = max(1, math.ceil(duration / chunk_size_seconds))
    return [
        AudioChunk(
            index=i,
            path=source_path,
            start_seconds=i * chunk_size_seconds,
            end_seconds=min((i + 1) * chunk_size_seconds, duration),
            duration_seconds=min((i + 1) * chunk_size_seconds, duration)
            - (i * chunk_size_seconds),
            temporary=False,
        )
        for i in range(chunk_count)
    ]


def materialize_audio_chunks(
    audio_path: str | Path,
    output_dir: str | Path,
    chunk_size_seconds: float = 10.0,
) -> List[AudioChunk]:
    source_path = Path(audio_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {source_path}")
    planned = plan_audio_chunks(
        audio_path=source_path,
        chunk_size_seconds=chunk_size_seconds,
    )
    if len(planned) == 1:
        return planned

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    result: List[AudioChunk] = []
    suffix = source_path.suffix or ".audio"
    for chunk in planned:
        chunk_path = target_dir / f"{source_path.stem}.chunk-{chunk.index:04d}{suffix}"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{chunk.start_seconds:.6f}",
            "-t",
            f"{chunk.duration_seconds:.6f}",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "copy",
            str(chunk_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
        result.append(
            AudioChunk(
                index=chunk.index,
                path=chunk_path,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                duration_seconds=chunk.duration_seconds,
                temporary=True,
            )
        )
    return result
