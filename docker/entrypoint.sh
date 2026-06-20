#!/usr/bin/env bash
set -euo pipefail

cd /workspace/livepeer

command="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command}" in
  help|-h|--help)
    cat <<'EOF'
Livepeer + Roboflow media workbench

Commands:
  local-ingest [args...]         Run the localhost audio ingest WebSocket service
  vdo-bridge [args...]           Run the stock-extension-compatible VDO signaling bridge
  run-workflow WORKFLOW [args...] Run arbitrary Roboflow workflow JSON with runtime parameters
  run-session WORKFLOW [args...]  Run arbitrary workflow JSON as a persisted session
  test [pytest args...]          Run local tests
  shell                          Start bash

Examples:
  livepeer-poc local-ingest --host 0.0.0.0 --port 8876
  livepeer-poc vdo-bridge --host 0.0.0.0 --port 9443
  livepeer-poc run-workflow /path/to/workflow.json --runtime-param source=stream_av53zc79i
  livepeer-poc test -q
EOF
    ;;
  local-ingest)
    exec python3 scripts/run_local_audio_ingest_server.py "$@"
    ;;
  vdo-bridge)
    exec python3 scripts/run_vdo_signaling_bridge.py "$@"
    ;;
  run-workflow)
    workflow_json="${1:-}"
    if [[ -z "${workflow_json}" ]]; then
      echo "usage: livepeer-poc run-workflow WORKFLOW_JSON [runtime args...]" >&2
      exit 2
    fi
    shift
    exec python3 scripts/run_livepeer_workflow_runtime.py "${workflow_json}" "$@"
    ;;
  run-session)
    workflow_json="${1:-}"
    if [[ -z "${workflow_json}" ]]; then
      echo "usage: livepeer-poc run-session WORKFLOW_JSON [runtime args...]" >&2
      exit 2
    fi
    shift
    exec python3 scripts/run_workflow_session.py "${workflow_json}" "$@"
    ;;
  test)
    if [[ $# -eq 0 ]]; then
      exec python3 -m pytest tests
    fi
    exec python3 -m pytest tests "$@"
    ;;
  shell)
    exec /bin/bash
    ;;
  *)
    exec "${command}" "$@"
    ;;
esac
