#!/usr/bin/env python3
"""Minimal standalone smoke test for the LOC-brokered audio transcription path.

This intentionally exercises the clearinghouse handoff flow used by this repo's
remote audio path. It does not add a separate direct gateway mode because that
is a different front door with different credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO_PATH = REPO_ROOT / "Why I Dont Make Fun Of Python.mp3"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "loc-audio-smoke"
DEFAULT_TARGET_AUDIO_DURATION_SECONDS = 81.893878

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboflow_livepeer_blocks.audio import AudioChunk, probe_audio_duration_seconds
from roboflow_livepeer_blocks.client import LivepeerOpenClearinghouseClient
from roboflow_livepeer_blocks.config import (
    DEFAULT_OPEN_CLEARINGHOUSE_URL,
    OPEN_CLEARINGHOUSE_API_KEY_ENV,
    OPEN_CLEARINGHOUSE_URL_ENV,
)


def utc_run_id(prefix: str = "loc-audio-smoke") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_success(result: Dict[str, Any]) -> None:
    if not str(result.get("text") or "").strip():
        raise RuntimeError("Transcription returned empty text")
    if int(result.get("actual_units") or 0) <= 0:
        raise RuntimeError("Transcription did not report positive actual_units")
    if not result.get("chunks"):
        raise RuntimeError("Transcription did not return chunk metadata")


def resolve_duration_seconds(audio_path: Path, explicit_duration: float | None) -> float:
    if explicit_duration is not None:
        if explicit_duration <= 0:
            raise ValueError("--duration-seconds must be positive")
        return explicit_duration
    if audio_path.resolve() == DEFAULT_AUDIO_PATH.resolve():
        return DEFAULT_TARGET_AUDIO_DURATION_SECONDS
    return probe_audio_duration_seconds(audio_path)


def transcribe_single_shot(
    *,
    client: LivepeerOpenClearinghouseClient,
    audio_path: Path,
    duration_seconds: float,
) -> Dict[str, Any]:
    chunk = AudioChunk(
        index=0,
        path=audio_path,
        start_seconds=0.0,
        end_seconds=duration_seconds,
        duration_seconds=duration_seconds,
        temporary=False,
    )
    chunk_result = client.transcribe_chunk(chunk)
    return {
        "text": chunk_result.text,
        "chunks": [
            {
                "index": chunk_result.chunk.index,
                "start_seconds": chunk_result.chunk.start_seconds,
                "end_seconds": chunk_result.chunk.end_seconds,
                "duration_seconds": chunk_result.chunk.duration_seconds,
                "temporary": chunk_result.chunk.temporary,
                "audio_path": str(audio_path),
                "chunk_file_path": str(audio_path),
                "text": chunk_result.text,
                "actual_units": chunk_result.actual_units,
                "job_id": chunk_result.job_id,
                "work_id": chunk_result.work_id,
            }
        ],
        "actual_units": chunk_result.actual_units,
        "job_ids": [chunk_result.job_id] if chunk_result.job_id else [],
        "work_ids": [chunk_result.work_id] if chunk_result.work_id else [],
        "raw_responses": [chunk_result.raw_responses],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-path",
        default=str(DEFAULT_AUDIO_PATH),
        help="Audio file to transcribe. Defaults to the repo MP3 sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Artifact directory. Defaults to artifacts/loc-audio-smoke/<utc-run-id>/",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Send the file in one brokered request instead of 10s chunks.",
    )
    parser.add_argument(
        "--chunk-size-seconds",
        type=float,
        default=10.0,
        help="Chunk size for the default chunked path.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Explicit duration override, mainly for single-shot mode on non-default files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.is_file():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        return 2

    api_key = os.getenv(OPEN_CLEARINGHOUSE_API_KEY_ENV)
    if not api_key:
        print(f"Missing {OPEN_CLEARINGHOUSE_API_KEY_ENV}", file=sys.stderr)
        return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / utc_run_id()).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = os.getenv(OPEN_CLEARINGHOUSE_URL_ENV, DEFAULT_OPEN_CLEARINGHOUSE_URL)
    ffprobe_available = shutil.which("ffprobe") is not None
    ffmpeg_available = shutil.which("ffmpeg") is not None
    use_single_shot = bool(args.single_shot)
    fallback_reason = ""
    if not use_single_shot and (not ffprobe_available or not ffmpeg_available):
        use_single_shot = True
        fallback_reason = "ffmpeg/ffprobe unavailable; falling back to brokered single-shot mode"
    request_metadata = {
        "audio_path": str(audio_path),
        "audio_size_bytes": audio_path.stat().st_size,
        "base_url": base_url,
        "requested_mode": "single-shot" if args.single_shot else "chunked",
        "selected_mode": "single-shot" if use_single_shot else "chunked",
        "chunk_size_seconds": None if use_single_shot else args.chunk_size_seconds,
        "duration_seconds_override": args.duration_seconds,
        "has_api_key": True,
        "api_key_env_var": OPEN_CLEARINGHOUSE_API_KEY_ENV,
        "ffprobe_available": ffprobe_available,
        "ffmpeg_available": ffmpeg_available,
        "fallback_reason": fallback_reason,
        "run_id": output_dir.name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output_dir / "request.json", request_metadata)

    try:
        with LivepeerOpenClearinghouseClient(api_key=api_key, base_url=base_url) as client:
            if use_single_shot:
                result = transcribe_single_shot(
                    client=client,
                    audio_path=audio_path,
                    duration_seconds=resolve_duration_seconds(audio_path, args.duration_seconds),
                )
            else:
                result = client.transcribe_audio_file(
                    audio_path=str(audio_path),
                    chunk_size_seconds=args.chunk_size_seconds,
                )
        ensure_success(result)
    except Exception as error:
        failure_payload = {
            "error_type": error.__class__.__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "request": request_metadata,
        }
        write_json(output_dir / "error.json", failure_payload)
        print(f"Transcription failed. Diagnostics: {output_dir / 'error.json'}", file=sys.stderr)
        return 1

    write_json(output_dir / "result.json", result)
    (output_dir / "transcript.txt").write_text(str(result.get("text") or ""), encoding="utf-8")
    summary = {
        "audio_path": str(audio_path),
        "mode": request_metadata["selected_mode"],
        "chunk_count": len(result.get("chunks") or []),
        "actual_units": result.get("actual_units"),
        "job_ids": result.get("job_ids") or [],
        "work_ids": result.get("work_ids") or [],
        "result_json_path": str(output_dir / "result.json"),
        "transcript_path": str(output_dir / "transcript.txt"),
        "fallback_reason": fallback_reason,
    }
    write_json(output_dir / "summary.json", summary)

    preview = " ".join(str(result.get("text") or "").split())
    print(json.dumps(summary, indent=2))
    if preview:
        print(f"Transcript preview: {preview[:280]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
