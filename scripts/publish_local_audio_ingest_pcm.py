#!/usr/bin/env python3
"""Publish a local audio file into the localhost ingest WebSocket in realtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboflow_livepeer_blocks.local_ingest import parse_local_audio_ingest_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path")
    parser.add_argument(
        "--source",
        default="test-session",
        help="Ingest session id or ws://.../v1/ingest/audio/{session_id} URL.",
    )
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--frame-duration-seconds", type=float, default=0.08)
    parser.add_argument(
        "--startup-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay before the first audio frame is sent after the websocket opens.",
    )
    args = parser.parse_args()

    if args.duration_seconds <= 0:
        raise SystemExit("--duration-seconds must be positive")
    if args.sample_rate <= 0:
        raise SystemExit("--sample-rate must be positive")
    if args.channels != 1:
        raise SystemExit("only mono audio is supported")
    if args.frame_duration_seconds <= 0:
        raise SystemExit("--frame-duration-seconds must be positive")

    from websockets.sync.client import connect

    info = parse_local_audio_ingest_source(source=args.source)
    ingest_url = info["ingest_url"]
    frame_bytes = max(2, int(round(args.sample_rate * args.frame_duration_seconds)) * 2)
    target_bytes = int(round(args.duration_seconds * args.sample_rate)) * 2
    ffmpeg_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stream_loop",
        "-1",
        "-i",
        str(Path(args.audio_path).resolve()),
        "-f",
        "s16le",
        "-ar",
        str(args.sample_rate),
        "-ac",
        str(args.channels),
        "pipe:1",
    ]
    sent_bytes = 0
    process = subprocess.Popen(
        ffmpeg_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe was not created")
    with connect(ingest_url, open_timeout=20, close_timeout=5) as websocket:
        opened = websocket.recv(timeout=5)
        print(opened)
        if args.startup_delay_seconds > 0:
            time.sleep(args.startup_delay_seconds)
        while sent_bytes < target_bytes:
            started_at = time.monotonic()
            frame = process.stdout.read(frame_bytes)
            if not frame:
                raise RuntimeError("ffmpeg ended before duration_seconds target was reached")
            if len(frame) > target_bytes - sent_bytes:
                frame = frame[: target_bytes - sent_bytes]
            websocket.send(frame)
            sent_bytes += len(frame)
            elapsed = time.monotonic() - started_at
            sleep_seconds = max(0.0, args.frame_duration_seconds - elapsed)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        summary = {
            "ingest_url": ingest_url,
            "audio_path": str(Path(args.audio_path).resolve()),
            "duration_seconds": args.duration_seconds,
            "sent_bytes": sent_bytes,
            "sent_audio_seconds": round(sent_bytes / 2 / args.sample_rate, 3),
        }
        print(json.dumps(summary, sort_keys=True))
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
