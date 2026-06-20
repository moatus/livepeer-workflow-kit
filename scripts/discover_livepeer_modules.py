#!/usr/bin/env python3
"""Print the Cloudspe/Livepeer Modules capability catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboflow_livepeer_blocks.providers import require_livepeer_modules_provider


def main() -> int:
    provider = require_livepeer_modules_provider()
    catalog = provider.discover()
    summary = [
        {
            "capability": item.capability,
            "offering": item.offering,
            "work_unit": item.work_unit,
            "interaction_mode": item.interaction_mode,
            "extra": item.extra,
        }
        for item in catalog.offerings
    ]
    print(json.dumps({"provider": provider.name, "base_url": provider.base_url, "items": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
