# Livepeer Workflow Kit

Livepeer Workflow Kit is a framework for giving agents access to Livepeer-powered media services inside agentic workflows.

It lets an agent build and run workflows that capture media, transcribe audio, analyze visuals, extract text, and produce durable artifacts such as transcripts, frame summaries, event logs, and highlight clips.

The kit is designed to be self-owned: it can run from your machine, while GPU-heavy work such as transcription and vision analysis can be offloaded to Livepeer Modules through Cloudspe. Self-hosted runners are still possible, but remote Livepeer services are the default path.

## Example: Meeting Copilot

A meeting copilot built with this kit can run from your machine, watch a browser tab or call, and use Livepeer services for the expensive media intelligence work.

```text
browser tab or call
  -> Chrome capture extension
  -> local VDO bridge
  -> workflow session
  -> Livepeer transcription + vision modules
  -> local artifacts, summaries, and clips
```

The full stack can:

- capture a browser tab, webinar, call, demo, or screenshare through the Chrome extension
- ingest the live stream through the local bridge container
- resolve and preflight active streams before capture
- record bounded windows of audio and video
- transcribe speech with Livepeer audio transcription modules
- analyze screen frames with Livepeer vision modules
- extract visible text from slides, docs, IDEs, dashboards, and shared screens
- detect slide/frame changes and preserve sampled frames
- write structured events, transcripts, visual results, and status files
- summarize what happened using the transcript and visual artifacts
- cut short highlight clips from the captured media
- keep all workflow artifacts on your machine while offloading GPU-heavy inference

For browser tab capture, use this Chrome extension with the kit:

```text
https://chromewebstore.google.com/detail/vdoninja-video-capture/hppndmepdhaplfamkeblnhpjmiigcdij
```

The extension publishes the browser tab to the local bridge. The workflow then consumes that bridge stream.

## What Is Included

- `roboflow_livepeer_blocks/`: Livepeer media/source, transcription, visual-analysis, provider, and workflow-authoring blocks.
- `scripts/run_workflow_session.py`: generic persisted workflow session runner.
- `scripts/run_vdo_signaling_bridge.py`: VDO/WebRTC signaling bridge for browser-extension publishing.
- `skills/livepeer-workflow-construction/`: agent-facing workflow construction skill.
- Docker workbench for repeatable execution.

## Browser Capture

For browser/webinar capture, run the bridge and publish from the Chrome extension to:

```text
wss://localhost:9443
```

Host status check:

```bash
curl -sk https://localhost:9443/statusz
```

An active publisher should show `stream_count > 0`.

Inside Docker, use the compose service URL:

```text
wss://vdo-signaling-bridge:9443
```

## Docker Path

```bash
docker compose up -d vdo-signaling-bridge

docker compose run --rm \
  -v /absolute/host/run-dir:/workspace/authoring-test \
  -e LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY \
  -e LIVEPEER_OPEN_CLEARINGHOUSE_URL \
  livepeer-poc run-session /workspace/authoring-test/workflow.json \
  --run-id run-001 \
  --source-preflight wss://vdo-signaling-bridge:9443 \
  --runtime-param source=auto
```

## Native Path

Native install is supported when the host has Python, ffmpeg, GStreamer, PyGObject, and the required GStreamer plugins installed.

```bash
python3 -m pip install -e .
python3 -m pip install -r docker/requirements.workbench.txt
export PYTHONPATH="$PWD:$PWD/references/roboflow-inference"
export LIVEPEER_OPEN_CLEARINGHOUSE_URL=https://loc.cloudspe.com
export LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY=...
```

Run the bridge natively:

```bash
python3 scripts/run_vdo_signaling_bridge.py --host 0.0.0.0 --port 9443
```

Run a workflow session natively:

```bash
python3 scripts/run_workflow_session.py workflow.json \
  --run-id run-001 \
  --source-preflight wss://localhost:9443 \
  --runtime-param source=auto
```

Use `wss://localhost:9443` in workflow source settings for native execution.

## Provider Default

Livepeer Modules via Cloudspe is the default remote provider. Set:

```bash
LIVEPEER_OPEN_CLEARINGHOUSE_URL=https://loc.cloudspe.com
LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY=...
```

Self-hosted audio or vision runners are an explicit alternative mode, not the default.

## Validation

```bash
python3 -m compileall -q roboflow_livepeer_blocks scripts tests
python3 -m pytest -q tests/test_authoring_surface.py
bash -n docker/entrypoint.sh
docker compose config --quiet
```
