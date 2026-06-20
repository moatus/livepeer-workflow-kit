# Artifact Contract

Use session/run directories so workflow results are auditable.

## Top-Level Files

Recommended files in each run directory:

- `workflow.json`: materialized Roboflow workflow.
- `runtime-parameters.json`: runtime parameter values.
- `preflight.json`: capture source status/resolution.
- `events.jsonl`: session events.
- `workflow-result.json`: raw Roboflow execution result.
- `status.json`: final machine-readable status and artifact index.
- `summary.md` or `summary.json`: human-facing handoff summary.

## Status JSON

Useful fields:

```json
{
  "schema_version": "livepeer.workflow_run.v1",
  "status": "completed",
  "run_id": "example-run",
  "source": "stream_example",
  "signaling_server": "wss://vdo-signaling-bridge-tls:9443",
  "artifact_root": "/output/example-run",
  "workflow_json_path": "/output/example-run/workflow.json",
  "workflow_result_json_path": "/output/example-run/workflow-result.json",
  "events_jsonl_path": "/output/example-run/events.jsonl",
  "preflight": {"ok": true},
  "summary": {}
}
```

When failed, include:

- `error_type`
- `error`
- `failed_layer`: one of `source`, `workflow_json`, `runtime`, `audio`, `vision`, `artifact_inspection`, `unknown`

## Common Artifact Patterns

Transcript artifacts often live under:

```text
<artifact-root>/audio-true-streaming/*/true-streaming-transcript.txt
<artifact-root>/audio-true-streaming/*/true-streaming-session-result.json
<artifact-root>/audio-true-streaming/*/true-streaming-events.jsonl
```

Visual artifacts often live under:

```text
<artifact-root>/visual-analysis/*/result.json
<artifact-root>/visual-analysis/*/meeting-visual-events.jsonl
<artifact-root>/visual-analysis/*/frames/
<artifact-root>/visual-analysis/*/slides/
```

## Handoff Summary

Report:

- final status
- resolved source and bridge stream ID
- runtime command/script
- workflow path
- transcript path and short transcript summary
- visual result path and short visual summary
- notable failures or recovery paths
