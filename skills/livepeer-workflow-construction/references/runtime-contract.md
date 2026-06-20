# Runtime Contract

This reference describes generic execution for authored workflow JSON.

## Generic Docker Runtime

Use the Docker workbench session runner to run arbitrary authored workflow JSON and write session sidecars:

```bash
cd /work/repos/livepeer/livepeer-roboflow

docker compose run --rm \
  -v /absolute/host/test-dir:/workspace/authoring-test \
  livepeer-poc run-session /workspace/authoring-test/workflow.json \
  --run-id test-run \
  --source-preflight wss://vdo-signaling-bridge:9443 \
  --runtime-param source=<stream-id-or-auto>
```

Use `run-workflow` when you only need the raw workflow result:

```bash
cd /work/repos/livepeer/livepeer-roboflow

docker compose run --rm \
  -v /absolute/host/test-dir:/workspace/authoring-test \
  livepeer-poc run-workflow /workspace/authoring-test/workflow.json \
  --runtime-param source=<stream-id-or-auto> \
  --output-json /workspace/authoring-test/workflow-result.json
```

Notes:

- Mount a host test directory so `workflow.json`, `workflow-result.json`, logs, and output artifacts persist.
- The canonical container session root is `/workspace/authoring-test`.
- The canonical container artifact root is `/workspace/authoring-test/artifacts`.
- If `--output-root` is omitted, `run-session` uses `/workspace/authoring-test/artifacts`.
- When authoring workflow block `output_dir` values in Python, use `WorkflowBuilder().artifact_path("...")` so artifacts land under the mounted root.
- Runtime parameters use repeated `--runtime-param key=value`.
- If the workflow has `"$inputs.source"`, pass `--runtime-param source=<value>`.
- `run-session` writes `preflight.json`, `runtime-parameters.json`, `session-context.json`, `workflow-result.json`, `status.json`, and `summary.md`.
- For Livepeer Modules / Cloudspe execution, the container must receive `LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY`. Compose reads this from the environment or `.env`.
- `run-session` fails fast when a workflow selects Livepeer remote backends and the API key is missing.

## Host-Side Validation

Validate authored JSON before execution:

```bash
python3 -m json.tool /absolute/host/test-dir/workflow.json >/dev/null
```

Validate result/status JSON after execution:

```bash
python3 -m json.tool /absolute/host/test-dir/workflow-result.json >/dev/null
python3 -m json.tool /absolute/host/test-dir/status.json >/dev/null
```

## Runtime Environment URL Mapping

Browser/host URL values and container URL values can differ.

For stock browser-extension WSS publishing:

```text
host/browser bridge:  wss://localhost:9443
container bridge:     wss://vdo-signaling-bridge:9443
```

Use the container bridge URL in workflow JSON when the workflow runs in Docker. Some workbench variants expose the TLS bridge as `vdo-signaling-bridge-tls`; use the compose alias that resolves and returns `/statusz` in the current environment.

## Session Runner Responsibilities

`run-session` handles the common wrapper work:

1. Check source status and write `preflight.json` when `--source-preflight` is provided.
2. Validate and execute `workflow.json`.
3. Write `runtime-parameters.json`.
4. Write `session-context.json` with the container session/artifact roots.
5. Write `workflow-result.json`.
6. Write `status.json` with final status, artifact root, workflow path, result path, artifact index, and failure reason.
7. Write `summary.md` with a concise handoff scaffold.
