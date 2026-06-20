# Provider Discovery

Livepeer Modules through Cloudspe is the default provider.

The profile is a gateway adapter, not a hardcoded catalog. Discover capabilities from the provider API at authoring time, choose the best offering for the task, then pass the selected capability/offering into workflow blocks.

## Livepeer Modules / Cloudspe

Default URL:

```text
https://loc.cloudspe.com
```

Required environment:

```bash
LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY=...
```

If the user supplies the API key in the task prompt, export it into the process environment for authoring and Docker execution instead of hardcoding it into generated scripts or reports. For example:

```bash
export LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY='...'
export LIVEPEER_OPEN_CLEARINGHOUSE_URL='https://loc.cloudspe.com'
```

When running through Docker Compose, pass the environment through with `-e LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY -e LIVEPEER_OPEN_CLEARINGHOUSE_URL` or ensure the compose project `.env` provides those values. Avoid writing secrets into checked-in files, shell scripts, or handoff summaries.

Discovery uses:

```text
GET /v1/capabilities
X-API-Key: <key>
```

Route checks use:

```text
GET /v1/routes?capability=<capability>&offering=<offering>
X-API-Key: <key>
```

Do not use bearer auth for this provider. Use `X-API-Key`.

## Python Adapter

When a local `livepeer-roboflow/` repo copy is available, use the provider adapter:

```python
from roboflow_livepeer_blocks.providers import LivepeerModulesProvider

provider = LivepeerModulesProvider.from_env()

audio = provider.choose_audio_transcription(streaming=False)
provider.require_route(audio)
audio_params = provider.transcription_block_params(audio)

vision = provider.choose_screen_vision()
provider.require_route(vision)
vision_params = provider.vision_block_params(vision)
```

`audio_params` is suitable for `livepeer_true_streaming_transcription_session`.

`vision_params` is suitable for `livepeer_florence2_screen_slide_analysis`.

## Fast Fail

If Cloudspe credentials, discovery, or route selection fail, report the exact layer:

- missing `LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY`
- capability discovery failed
- no suitable capability/offering found
- no route available for the selected capability/offering
- workflow runtime failed after route selection

Do not silently fall back to local/self-hosted runners. Self-hosting is an explicit alternative mode, not the default.

## Local Development Alternative

For offline or self-hosted development, users may explicitly choose local runner URLs and local backends. That is a different provider mode and should be named in the workflow/report.
