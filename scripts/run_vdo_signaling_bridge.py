#!/usr/bin/env python3
"""Run the stock-extension-compatible VDO signaling bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from roboflow_livepeer_blocks.vdo_signaling_bridge import (
    DEFAULT_VDO_BRIDGE_CERT_HOSTS,
    app,
    ensure_vdo_bridge_certificate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the VDO signaling bridge over HTTPS/WSS.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument(
        "--cert-dir",
        default=str(Path("artifacts") / "vdo-signaling-bridge-certs"),
    )
    parser.add_argument(
        "--cert-host",
        action="append",
        default=[],
        help="Additional certificate SAN host or IP. May be provided multiple times.",
    )
    args = parser.parse_args()

    cert_dir = Path(args.cert_dir).resolve()
    cert_hosts = list(DEFAULT_VDO_BRIDGE_CERT_HOSTS) + list(args.cert_host)
    cert_path, key_path = ensure_vdo_bridge_certificate(
        cert_path=cert_dir / "bridge.crt",
        key_path=cert_dir / "bridge.key",
        hosts=cert_hosts,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
