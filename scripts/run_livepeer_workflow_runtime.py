#!/usr/bin/env python3
"""Run the Livepeer transcription workflow through Roboflow's execution engine."""

from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_REFERENCE = REPO_ROOT / "references" / "roboflow-inference"

for path in (REPO_ROOT, ROBOFLOW_REFERENCE):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def _parse_runtime_params(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_parameters: Dict[str, Any] = {}
    if args.audio_path:
        runtime_parameters["audio_path"] = str(Path(args.audio_path).resolve())
    if args.runtime_params_json:
        runtime_parameters.update(json.loads(args.runtime_params_json))
    if args.runtime_params_file:
        runtime_parameters.update(json.loads(Path(args.runtime_params_file).read_text()))
    for item in args.runtime_param:
        if "=" not in item:
            raise ValueError(f"runtime parameter must be KEY=VALUE: {item!r}")
        key, value = item.split("=", 1)
        runtime_parameters[key] = value
    return runtime_parameters


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Livepeer Roboflow workflow through Roboflow's execution engine."
    )
    parser.add_argument("workflow_json")
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Backwards-compatible shortcut that sets runtime parameter audio_path.",
    )
    parser.add_argument(
        "--runtime-params-json",
        help="JSON object merged into Roboflow workflow runtime parameters.",
    )
    parser.add_argument(
        "--runtime-params-file",
        help="Path to a JSON object merged into Roboflow workflow runtime parameters.",
    )
    parser.add_argument(
        "--runtime-param",
        action="append",
        default=[],
        help="Runtime parameter as KEY=VALUE. May be provided multiple times.",
    )
    parser.add_argument(
        "--require-text",
        action="store_true",
        help="Fail if the first workflow result has an empty text field.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path where the full Roboflow workflow result JSON is written.",
    )
    args = parser.parse_args()

    os.environ.setdefault("WORKFLOWS_PLUGIN_ONLY_CORE", "true")
    os.environ.setdefault("WORKFLOWS_PLUGINS", "roboflow_livepeer_blocks")

    from inference.core.workflows.execution_engine.core import ExecutionEngine

    workflow_path = Path(args.workflow_json)
    workflow = json.loads(workflow_path.read_text())
    runtime_parameters = _parse_runtime_params(args)
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={},
        max_concurrent_steps=1,
    )
    result = engine.run(runtime_parameters=runtime_parameters)
    print(json.dumps(result, indent=2))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    text = result[0].get("text", "") if result else ""
    if args.require_text and not text.strip():
        raise RuntimeError("Workflow returned empty text")
    if text.strip():
        print("WORKFLOW_TEXT_START")
        print(text)
        print("WORKFLOW_TEXT_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
