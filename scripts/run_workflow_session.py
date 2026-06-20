#!/usr/bin/env python3
"""Run any authored Livepeer Roboflow workflow as a persisted session."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_REFERENCE = REPO_ROOT / "references" / "roboflow-inference"

for path in (REPO_ROOT, ROBOFLOW_REFERENCE):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def _parse_runtime_params(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_parameters: Dict[str, Any] = {}
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


def _statusz_url(source: str) -> str:
    base = source.rstrip("/")
    if base.startswith("wss://"):
        base = "https://" + base[len("wss://") :]
    elif base.startswith("ws://"):
        base = "http://" + base[len("ws://") :]
    if not base.endswith("/statusz"):
        base = base + "/statusz"
    return base


def _preflight(source: Optional[str], timeout_seconds: float) -> Dict[str, Any]:
    if not source:
        return {"status": "skipped", "reason": "no source preflight URL provided"}
    url = _statusz_url(source)
    started = time.time()
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds, context=context) as response:
            body = response.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        return {
            "status": "ok",
            "url": url,
            "elapsed_seconds": round(time.time() - started, 3),
            "payload": payload,
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {
            "status": "failed",
            "url": url,
            "elapsed_seconds": round(time.time() - started, 3),
            "error": str(exc),
        }


def _find_first(root: Path, patterns: Iterable[str]) -> Optional[str]:
    if not root.exists():
        return None
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return str(matches[0])
    return None


def _artifact_index(output_root: Path) -> Dict[str, Any]:
    transcript = _find_first(
        output_root,
        [
            "true-streaming-transcript.txt",
            "direct-true-streaming-transcript.txt",
            "*transcript.txt",
        ],
    )
    visual_result = _find_first(output_root, ["visual-analysis/*/result.json", "result.json"])
    event_logs = [str(path) for path in sorted(output_root.rglob("*.jsonl"))]
    frames = [str(path) for path in sorted(output_root.rglob("frames/*.jpg"))]
    slides = [str(path) for path in sorted(output_root.rglob("slides/*.jpg"))]
    return {
        "transcript_path": transcript,
        "visual_result_path": visual_result,
        "event_logs": event_logs,
        "frame_count": len(frames),
        "slide_count": len(slides),
        "sample_frames": frames[:5],
        "sample_slides": slides[:5],
    }


def _preview_text(path: Optional[str], limit: int = 1200) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _visual_status(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "status": data.get("status"),
        "frame_count": data.get("frame_count"),
        "slide_count": data.get("slide_count"),
        "summary": data.get("meeting_visual_summary"),
    }


def _requires_livepeer_modules(workflow: Dict[str, Any]) -> bool:
    for step in workflow.get("steps", []):
        if not isinstance(step, dict):
            continue
        if step.get("transcription_backend") in {"livepeer_remote", "livepeer_remote_http"}:
            return True
        if step.get("vision_backend") == "livepeer_remote":
            return True
    return False


def _write_summary(path: Path, status: Dict[str, Any]) -> None:
    artifacts = status.get("artifacts", {})
    transcript_preview = _preview_text(artifacts.get("transcript_path"))
    visual = _visual_status(artifacts.get("visual_result_path"))
    lines = [
        f"# Workflow Session {status['run_id']}",
        "",
        f"- Status: `{status['status']}`",
        f"- Workflow: `{status['workflow_path']}`",
        f"- Output root: `{status['output_root']}`",
        f"- Result JSON: `{status.get('workflow_result_path') or ''}`",
        f"- Transcript: `{artifacts.get('transcript_path') or ''}`",
        f"- Visual result: `{artifacts.get('visual_result_path') or ''}`",
        f"- Frames: `{artifacts.get('frame_count', 0)}`",
        f"- Slides: `{artifacts.get('slide_count', 0)}`",
    ]
    if status.get("error"):
        lines.extend(["", "## Error", "", "```text", str(status["error"]), "```"])
    if transcript_preview:
        lines.extend(["", "## Transcript Preview", "", transcript_preview])
    if visual:
        lines.extend(["", "## Visual Status", "", "```json", json.dumps(visual, indent=2), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an authored Livepeer Roboflow workflow and persist session artifacts."
    )
    parser.add_argument("workflow_json")
    parser.add_argument("--run-id", default="", help="Stable session/run identifier.")
    parser.add_argument(
        "--output-root",
        help=(
            "Container-visible directory for artifacts and sidecars. Defaults to "
            "the canonical persisted artifact root /workspace/authoring-test/artifacts."
        ),
    )
    parser.add_argument("--source-preflight", help="Optional source/bridge URL to preflight.")
    parser.add_argument("--preflight-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--runtime-params-json", help="JSON object of runtime parameters.")
    parser.add_argument("--runtime-params-file", help="JSON file of runtime parameters.")
    parser.add_argument(
        "--runtime-param",
        action="append",
        default=[],
        help="Runtime parameter as KEY=VALUE. May be provided multiple times.",
    )
    parser.add_argument("--workflow-result-json", help="Path for raw workflow result JSON.")
    parser.add_argument("--status-json", help="Path for session status JSON.")
    parser.add_argument("--summary-md", help="Path for human-readable session summary.")
    args = parser.parse_args()

    os.environ.setdefault("WORKFLOWS_PLUGIN_ONLY_CORE", "true")
    os.environ.setdefault("WORKFLOWS_PLUGINS", "roboflow_livepeer_blocks")

    from inference.core.workflows.execution_engine.core import ExecutionEngine
    from roboflow_livepeer_blocks.authoring import (
        ARTIFACT_ROOT_ENV,
        DEFAULT_ARTIFACT_ROOT,
        DEFAULT_SESSION_ROOT,
        SESSION_ROOT_ENV,
        load_workflow,
    )

    workflow_path = Path(args.workflow_json)
    output_root = Path(args.output_root or os.getenv(ARTIFACT_ROOT_ENV, DEFAULT_ARTIFACT_ROOT))
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(SESSION_ROOT_ENV, DEFAULT_SESSION_ROOT)
    os.environ.setdefault(ARTIFACT_ROOT_ENV, str(output_root))
    run_id = args.run_id or output_root.name
    result_path = Path(args.workflow_result_json) if args.workflow_result_json else output_root / "workflow-result.json"
    status_path = Path(args.status_json) if args.status_json else output_root / "status.json"
    summary_path = Path(args.summary_md) if args.summary_md else output_root / "summary.md"

    started = time.time()
    preflight = _preflight(args.source_preflight, args.preflight_timeout_seconds)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    session_context = {
        "container_session_root": os.environ[SESSION_ROOT_ENV],
        "container_artifact_root": os.environ[ARTIFACT_ROOT_ENV],
        "artifact_root_env": ARTIFACT_ROOT_ENV,
        "session_root_env": SESSION_ROOT_ENV,
        "workflow_path": str(workflow_path),
    }
    (output_root / "session-context.json").write_text(
        json.dumps(session_context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime_parameters = _parse_runtime_params(args)
    (output_root / "runtime-parameters.json").write_text(
        json.dumps(runtime_parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    status: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "workflow_path": str(workflow_path),
        "output_root": str(output_root),
        "workflow_result_path": str(result_path),
        "status_path": str(status_path),
        "summary_path": str(summary_path),
        "session_context": session_context,
        "preflight": preflight,
        "runtime_parameters": runtime_parameters,
        "started_at_unix": started,
    }

    exit_code = 0
    try:
        workflow = load_workflow(workflow_path)
        if _requires_livepeer_modules(workflow) and not os.getenv("LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY"):
            raise RuntimeError(
                "Workflow selected Livepeer Modules/Cloudspe remote execution, but "
                "LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY is not set. Set that key to use "
                "https://loc.cloudspe.com, or explicitly author a self-hosted/local "
                "provider workflow for development."
            )
        engine = ExecutionEngine.init(
            workflow_definition=workflow,
            init_parameters={},
            max_concurrent_steps=1,
        )
        result = engine.run(runtime_parameters=runtime_parameters)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        status["status"] = "completed"
        status["result_count"] = len(result) if isinstance(result, list) else None
        print(json.dumps(result, indent=2))
        text = result[0].get("text", "") if isinstance(result, list) and result else ""
        if isinstance(text, str) and text.strip():
            print("WORKFLOW_TEXT_START")
            print(text)
            print("WORKFLOW_TEXT_END")
    except Exception as exc:  # noqa: BLE001 - the session status should capture any failure.
        exit_code = 1
        status["status"] = "failed"
        status["error"] = str(exc)
        status["traceback"] = traceback.format_exc()
        print(status["traceback"], file=sys.stderr)
    finally:
        status["finished_at_unix"] = time.time()
        status["elapsed_seconds"] = round(status["finished_at_unix"] - started, 3)
        status["artifacts"] = _artifact_index(output_root)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary(summary_path, status)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
