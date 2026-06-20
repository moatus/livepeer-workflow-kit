import json
from pathlib import Path

from roboflow_livepeer_blocks.block import (
    LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1,
)
from roboflow_livepeer_blocks.livepeer_http_chunking_client import (
    LivepeerRemoteFallbackTransportClient,
    LivepeerRemoteHttpChunkingClient,
)
from roboflow_livepeer_blocks.livepeer_realtime_client import (
    LivepeerRemoteTrueStreamingWebSocketClient,
)
from roboflow_livepeer_blocks.true_streaming import _true_streaming_result_payload


class FakeResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(body) if isinstance(body, (dict, list)) else str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeRealtimeHttpClient:
    def __init__(self):
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url.endswith("/v1/capabilities"):
            return FakeResponse(
                {
                    "items": [
                        {
                            "name": "openai:audio-transcriptions",
                            "work_unit": "seconds",
                            "offerings": [
                                {
                                    "id": "nemo-meeting-stream",
                                    "work_unit": "seconds",
                                    "extra": {
                                        "interaction_mode": "ws-realtime@v0",
                                        "streaming": {"sample_rate": 16000},
                                    },
                                }
                            ],
                        }
                    ]
                }
            )
        raise AssertionError(url)

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if url.endswith("/v1/sessions"):
            return FakeResponse(
                {
                    "session_id": "loc-session-1",
                    "work_id": "work-1",
                    "broker_url": "https://broker.example",
                    "mode": "ws-realtime@v0",
                    "payment_envelope": "pay",
                    "expected_value_wei": 10,
                    "funded_value_wei": 20,
                    "refill_endpoint": "/v1/sessions/loc-session-1/refill",
                    "close_endpoint": "/v1/sessions/loc-session-1/close",
                    "opened_at": "2026-06-11T17:00:00Z",
                }
            )
        if url.endswith("/close"):
            return FakeResponse(
                {
                    "session_id": "loc-session-1",
                    "work_id": "work-1",
                    "actual_units": kwargs["json"]["actual_units"],
                    "billed_value_wei": 10,
                    "refund_wei": 10,
                    "outcome": kwargs["json"].get("outcome", ""),
                    "closed_at": "2026-06-11T17:01:00Z",
                }
            )
        raise AssertionError(url)


class FakeLocChunkingHttpClient:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if url.endswith("/v1/jobs"):
            return FakeResponse(
                {
                    "job_id": "job-fallback-1",
                    "work_id": "work-fallback-1",
                    "broker_url": "https://fallback-broker.example",
                    "mode": "handoff",
                    "payment_envelope": "fallback-pay",
                    "settle_endpoint": "/v1/jobs/job-fallback-1/settle",
                }
            )
        if url == "https://fallback-broker.example/v1/cap":
            return FakeResponse(
                {"text": "fallback hello"},
                headers={"Livepeer-Work-Units": "1"},
            )
        if url.endswith("/settle"):
            return FakeResponse({"ok": True})
        raise AssertionError(url)


class FakeWebSocket:
    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout=None):
        if self._responses:
            return self._responses.pop(0)
        raise TimeoutError(f"timed out after {timeout}")


def test_livepeer_remote_true_streaming_client_opens_ws_session_and_closes_loc_session():
    fake_http = FakeRealtimeHttpClient()
    fake_ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "event_type": "session.snapshot",
                    "session_id": "stream_remote_1",
                    "status": "active",
                    "duration_seconds": 0.0,
                }
            ),
            json.dumps(
                {
                    "event_type": "transcript.segment",
                    "session_id": "stream_remote_1",
                    "text": "hello",
                    "speaker": "speaker_0",
                    "start": 0.0,
                    "end": 0.1,
                    "duration_seconds": 0.1,
                    "is_final": True,
                }
            ),
            json.dumps(
                {
                    "event_type": "transcript.session.finished",
                    "session_id": "stream_remote_1",
                    "duration_seconds": 0.1,
                    "is_final": True,
                }
            ),
        ]
    )
    captured_connect = {}

    def fake_connect(url, **kwargs):
        captured_connect["url"] = url
        captured_connect["kwargs"] = kwargs
        return fake_ws

    client = LivepeerRemoteTrueStreamingWebSocketClient(
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        offering="nemo-meeting-stream",
        estimated_runway_units=105,
        max_total_units=135,
        http_client=fake_http,
        websocket_connect=fake_connect,
    )

    with client.connect_session(
        session_id="requested-session",
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
        frame_duration_seconds=0.08,
    ) as session:
        emitted = session.send_audio_frame(b"\x00\x00" * 1600, source_event={"frame_index": 0})
        assert any(event["event_type"] == "transcript.segment" for event in emitted)
        finished = session.finish()

    assert any(event["event_type"] == "payment.session.opened" for event in session.events)
    assert any(event["event_type"] == "payment.session.closed" for event in session.events)
    assert any(event["event_type"] == "transcript.session.finished" for event in session.events)
    assert any(event["event_type"] == "payment.session.closed" for event in finished)
    assert captured_connect["url"].startswith("wss://broker.example/?")
    assert "session_id=loc-session-1" in captured_connect["url"]
    assert captured_connect["kwargs"]["additional_headers"]["Livepeer-Payment"] == "pay"
    assert fake_http.calls[1][2]["json"] == {
        "capability": "openai:audio-transcriptions",
        "offering": "nemo-meeting-stream",
        "estimated_runway_units": 105,
        "max_total_units": 135,
    }
    assert fake_http.calls[2][2]["json"]["actual_units"] == 1
    assert fake_ws.sent[-1] == json.dumps({"type": "finish"})
    prepared = next(
        event for event in session.events if event["event_type"] == "livepeer.realtime.websocket.handshake.prepared"
    )
    assert prepared["requested_session_id"] == "requested-session"
    assert prepared["billing_session_id"] == "loc-session-1"


def test_livepeer_remote_true_streaming_client_fails_fast_with_handshake_diagnostics():
    fake_http = FakeRealtimeHttpClient()
    attempted = {}

    def fake_connect(url, **kwargs):
        attempted["url"] = url
        attempted["kwargs"] = kwargs
        raise RuntimeError("EOF before HTTP response")

    client = LivepeerRemoteTrueStreamingWebSocketClient(
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        offering="nemo-meeting-stream",
        http_client=fake_http,
        websocket_connect=fake_connect,
    )

    try:
        with client.connect_session(session_id="requested-session") as session:
            session.finish()
    except RuntimeError as error:
        assert (
            str(error)
            == "Livepeer realtime websocket handshake failed; error=EOF before HTTP response"
        )
    else:
        raise AssertionError("Expected realtime handshake to fail")

    assert attempted["url"] == (
        "wss://broker.example/?session_id=loc-session-1&language=en"
        "&preset=meeting&max_speakers=4&sample_rate=16000"
    )
    assert attempted["kwargs"]["additional_headers"]["Livepeer-Payment"] == "pay"


def test_livepeer_remote_fallback_transport_uses_http_chunking_after_ws_handshake_failure():
    realtime_http = FakeRealtimeHttpClient()
    chunk_http = FakeLocChunkingHttpClient()
    attempted = {}

    def fake_connect(url, **kwargs):
        attempted["url"] = url
        attempted["kwargs"] = kwargs
        raise RuntimeError("EOF before HTTP response")

    client = LivepeerRemoteFallbackTransportClient(
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        realtime_offering="nemo-meeting-stream",
        estimated_runway_units=30,
        max_total_units=60,
        chunk_size_seconds=30.0,
        websocket_connect=fake_connect,
        realtime_http_client=realtime_http,
        chunk_http_client=chunk_http,
    )

    with client.connect_session(
        session_id="requested-session",
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
        frame_duration_seconds=0.08,
    ) as session:
        assert session.send_audio_frame(b"\x00\x00" * 1600) == []
        finished = session.finish()

    event_types = [event["event_type"] for event in session.events]
    assert "livepeer.realtime.websocket.handshake.failed" in event_types
    assert "transcript.transport.fallback" in event_types
    assert "payment.session.closed" in event_types
    assert "transcript.segment" in event_types
    assert "transcript.session.finished" in event_types
    assert any(event.get("text") == "fallback hello" for event in finished)
    assert attempted["url"].startswith("wss://broker.example/?")
    assert realtime_http.calls[2][2]["json"] == {
        "actual_units": 0,
        "outcome": "ws_handshake_failed_fallback_to_http_chunking",
    }
    assert [call[1] for call in chunk_http.calls] == [
        "https://loc.cloudspe.com/v1/jobs",
        "https://fallback-broker.example/v1/cap",
        "https://loc.cloudspe.com/v1/jobs/job-fallback-1/settle",
    ]
    assert chunk_http.calls[1][2]["data"] == {
        "model": "nemo-meeting",
        "response_format": "json",
    }
    assert "file" in chunk_http.calls[1][2]["files"]


def test_local_audio_ingest_block_selects_livepeer_remote_backend(monkeypatch):
    class FakeRunner:
        def run(self):
            return {
                "session_id": "stream_remote_1",
                "stream_id": "test-session",
                "status": "closed",
                "source_mode": "localhost_ingest_pcm",
                "source": "ws://127.0.0.1:8876/v1/ingest/audio/test-session",
                "sent_audio_seconds": 105.0,
                "sent_frame_count": 1313,
                "text": "hello remote",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 105.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "payment.session.closed"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
                "publisher_log_path": "",
                "ffmpeg_log_path": "",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_local_audio_ingest_true_streaming_runner",
        fake_build_runner,
    )
    block = LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1(
        runner_url="http://audio-diarized-transcription-runner:8080",
        local_audio_ingest_url="http://local-audio-ingest:8876",
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "localhost_ingest",
            "source": "ws://127.0.0.1:8876/v1/ingest/audio/test-session",
            "stream_id": "test-session",
            "status_url": "http://127.0.0.1:8876/v1/ingest/audio/test-session",
            "consume_url": "ws://127.0.0.1:8876/v1/ingest/audio/test-session/consume",
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=105,
        transcription_backend="livepeer_remote",
        livepeer_estimated_runway_units=105,
        livepeer_max_total_units=135,
    )

    assert captured["runner_url"] == "https://loc.cloudspe.com"
    assert captured["client_cls"] is LivepeerRemoteFallbackTransportClient
    assert captured["client_init_kwargs"] == {
        "api_key": "test-key",
        "capability": "openai:audio-transcriptions",
        "realtime_offering": "nemo-meeting-stream",
        "estimated_runway_units": 105,
        "max_total_units": 135,
    }
    assert result["text"] == "hello remote"


def test_local_audio_ingest_block_selects_explicit_livepeer_remote_http_backend(monkeypatch):
    class FakeRunner:
        def run(self):
            return {
                "session_id": "stream_remote_http_1",
                "stream_id": "test-session",
                "status": "closed",
                "source_mode": "localhost_ingest_pcm",
                "source": "ws://127.0.0.1:8876/v1/ingest/audio/test-session",
                "sent_audio_seconds": 30.0,
                "sent_frame_count": 375,
                "text": "hello remote http",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 30.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "payment.job.completed"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
                "publisher_log_path": "",
                "ffmpeg_log_path": "",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_local_audio_ingest_true_streaming_runner",
        fake_build_runner,
    )
    block = LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1(
        runner_url="http://audio-diarized-transcription-runner:8080",
        local_audio_ingest_url="http://local-audio-ingest:8876",
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "localhost_ingest",
            "source": "ws://127.0.0.1:8876/v1/ingest/audio/test-session",
            "stream_id": "test-session",
            "status_url": "http://127.0.0.1:8876/v1/ingest/audio/test-session",
            "consume_url": "ws://127.0.0.1:8876/v1/ingest/audio/test-session/consume",
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=30,
        transcription_backend="livepeer_remote_http",
    )

    assert captured["runner_url"] == "https://loc.cloudspe.com"
    assert captured["client_cls"] is LivepeerRemoteHttpChunkingClient
    assert captured["client_init_kwargs"] == {
        "api_key": "test-key",
        "capability": "openai:audio-transcriptions",
        "offering": "nemo-meeting",
    }
    assert result["text"] == "hello remote http"


def test_true_streaming_result_prefers_remote_event_session_id(tmp_path):
    result = _true_streaming_result_payload(
        session_id="requested-session",
        stream_id="test-session",
        captured_segments=[],
        events=[
            {"event_type": "payment.session.opened"},
            {
                "event_type": "transcript.segment",
                "session_id": "stream_remote_1",
                "text": "hello",
                "speaker": "speaker_0",
                "start": 0.0,
                "end": 1.0,
                "is_final": True,
            },
        ],
        events_path=tmp_path / "events.jsonl",
        result_path=tmp_path / "result.json",
        transcript_text_path=tmp_path / "transcript.txt",
    )

    assert result["session_id"] == "stream_remote_1"
    assert result["text"] == "hello"


def test_true_streaming_result_falls_back_to_consolidated_provisionals_when_finals_are_sparse(
    tmp_path,
):
    result = _true_streaming_result_payload(
        session_id="requested-session",
        stream_id="test-session",
        captured_segments=[],
        events=[
            {
                "event_type": "transcript.segment",
                "session_id": "stream_remote_1",
                "start": 0.0,
                "end": 1.0,
                "speaker": "speaker_0",
                "text": "hello",
                "is_final": False,
            },
            {
                "event_type": "transcript.segment",
                "session_id": "stream_remote_1",
                "start": 1.0,
                "end": 2.0,
                "speaker": "speaker_1",
                "text": "world",
                "is_final": False,
            },
            {
                "event_type": "transcript.segment",
                "session_id": "stream_remote_1",
                "start": 1.96,
                "end": 2.0,
                "speaker": "speaker_1",
                "text": "tail",
                "is_final": True,
            },
            {
                "event_type": "transcript.session.finished",
                "session_id": "stream_remote_1",
                "is_final": True,
            },
        ],
        events_path=tmp_path / "events.jsonl",
        result_path=tmp_path / "result.json",
        transcript_text_path=tmp_path / "transcript.txt",
    )

    assert result["text"] == "hello world tail"
