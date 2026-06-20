# Source Diagnostics

Use source diagnostics before building or running long workflows.

## Source Modes

- Explicit source: a raw VDO stream ID or viewer URL containing a stream/view parameter.
- Auto discovery: `auto`, empty source, or a bare bridge URL. Auto discovery means "find a registered publisher stream on the configured bridge."

## Status URL Mapping

Map signaling URLs to bridge status URLs:

```text
wss://host:port -> https://host:port/statusz
ws://host:port  -> http://host:port/statusz
```

For local browser publishing:

```text
browser/host WSS bridge:  wss://localhost:9443
browser/host status URL:  https://localhost:9443/statusz
container WSS bridge:     wss://vdo-signaling-bridge:9443
```

For local non-browser/plain WebSocket clients:

```text
host WS bridge:       ws://localhost:9080
host status URL:      http://localhost:9080/statusz
container WS bridge:  ws://vdo-signaling-bridge:9080
```

The stock browser extension uses WSS custom signaling servers. Use the TLS bridge for stock browser-extension publishing.

## Browser Capture Setup

For a browser tab, webinar, or screenshare source, the local ingest bridge complements the browser extension. The extension publishes the tab/call into the bridge; the workflow consumes the bridge stream.

Chrome extension:

```text
https://chromewebstore.google.com/detail/vdoninja-video-capture/hppndmepdhaplfamkeblnhpjmiigcdij
```

Two things must be true before a workflow can capture media:

1. The VDO signaling bridge container must be running and reachable on the host.
2. The browser must publish to that bridge, typically through the Chrome extension with its custom signaling server set to `wss://localhost:9443`.

If the user wants to capture a browser tab but has not installed or enabled the extension, tell them to install the extension above and set its custom signaling server to `wss://localhost:9443`.

The workflow does not capture directly from Chrome. It captures a stream registered on the bridge. If the user says a browser/webinar stream is running, verify the bridge state rather than assuming the extension is connected.

Host check:

```bash
curl -sk https://localhost:9443/statusz
```

Expected for an active browser publisher:

```text
client_count > 0
stream_count > 0
streams contains a stream id
```

If `stream_count` is zero, the usual causes are:

- the bridge container is not running or the wrong bridge instance is being checked
- the Chrome extension is not installed, not enabled, or not publishing
- the extension is pointed at a different custom signaling server
- the browser tab/share was restarted after the bridge changed and needs to republish

Inside Docker, use the compose service alias that resolves in the workbench network. Common aliases are:

```text
wss://vdo-signaling-bridge:9443
wss://vdo-signaling-bridge-tls:9443
```

Prefer the alias that successfully answers `/statusz` from the container. Do not use `host.docker.internal` unless the environment explicitly supports it.

## Status Interpretation

Bridge status should include nonzero clients and streams before capture starts.

Useful fields:

- `client_count`
- `stream_count`
- `streams`
- `clients[].stream_id`

If `stream_count` is zero, there is no usable publisher on that bridge.

If a bridge stream ID has an extra suffix, workflow source resolution may normalize it to a playable stream ID. Report both the bridge stream ID and the runtime source when available.

## Preflight Output Contract

Write preflight results as JSON when possible:

```json
{
  "ok": true,
  "requested_source": "auto",
  "signaling_server": "wss://vdo-signaling-bridge:9443",
  "status_url": "https://localhost:9443/statusz",
  "bridge_stream_id": "stream_example808d64",
  "source": "stream_example"
}
```
