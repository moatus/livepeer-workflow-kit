#!/usr/bin/env python3
"""Run the local audio ingest WebSocket service."""

from __future__ import annotations

import argparse

import uvicorn

from roboflow_livepeer_blocks.local_ingest import app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the localhost audio ingest service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
