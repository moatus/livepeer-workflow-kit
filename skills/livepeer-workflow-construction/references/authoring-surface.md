# Authoring Surface

Use this reference when a local `livepeer-roboflow/` repo copy is available.

## Generic Builder

Use `roboflow_livepeer_blocks.authoring.WorkflowBuilder` to construct workflow JSON while still choosing the graph yourself.

The builder knows workflow mechanics:

- workflow inputs
- step dictionaries
- `$inputs.*` and `$steps.*.*` references
- `JsonField` outputs
- duplicate-name checks
- JSON writing

The builder does not choose a task topology.

```python
from roboflow_livepeer_blocks.authoring import WorkflowBuilder

w = WorkflowBuilder()
source = w.input("source")

media = w.step(
    "media_source",
    "roboflow_livepeer_blocks/livepeer_vdo_ninja_media_source@v1",
    source=source,
    signaling_server="wss://vdo-signaling-bridge:9443",
    audio_enabled=True,
    video_enabled=True,
)

pcm = w.step(
    "pcm16",
    "roboflow_livepeer_blocks/livepeer_pcm16_audio_transform@v1",
    source_descriptor=media.output("audio_source_descriptor"),
)

w.output("stream_id", media.output("stream_id"))
w.output("pcm_descriptor", pcm.output("pcm_descriptor"))
w.write_json("workflow.json")
```

Use `references/raw-block-catalog.md` to choose block types, inputs, and useful output selectors.

For block artifact paths, use the builder's canonical container artifact root:

```python
audio_output_dir = w.artifact_path("audio-true-streaming")
visual_output_dir = w.artifact_path("visual-analysis")
```

This resolves under `/workspace/authoring-test/artifacts` by default, which is mounted to the host in the Docker runtime examples. Do not bake host paths such as `/work/repos/...` into workflow block `output_dir` values.

Use `references/provider-discovery.md` to discover Livepeer Modules capabilities and get provider-derived block params instead of hardcoding offerings.

Example:

```python
from roboflow_livepeer_blocks.authoring import WorkflowBuilder
from roboflow_livepeer_blocks.providers import LivepeerModulesProvider

provider = LivepeerModulesProvider.from_env()
audio_capability = provider.choose_audio_transcription(streaming=False)
provider.require_route(audio_capability)
audio_params = provider.transcription_block_params(audio_capability)

vision_capability = provider.choose_screen_vision()
provider.require_route(vision_capability)
vision_params = provider.vision_block_params(vision_capability)

w = WorkflowBuilder()
# choose and wire blocks, then pass **audio_params / **vision_params into the relevant steps
```

## Generic Session Runner

Use `livepeer-poc run-session` to run any authored workflow JSON as a persisted session.

```bash
cd livepeer-roboflow

docker compose run --rm \
  -v /absolute/host/run-dir:/workspace/authoring-test \
  livepeer-poc run-session /workspace/authoring-test/workflow.json \
  --run-id my-run \
  --source-preflight wss://vdo-signaling-bridge:9443 \
  --runtime-param source=auto
```

The runner writes:

- `preflight.json`
- `runtime-parameters.json`
- `session-context.json`
- `workflow-result.json`
- `status.json`
- `summary.md`

It also indexes common transcript, visual result, event, frame, and slide artifacts when they appear under `--output-root`.

Use `run-workflow` directly only when you need the lowest-level runtime behavior.
