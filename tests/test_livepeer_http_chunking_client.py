import json

import pytest

from roboflow_livepeer_blocks.livepeer_http_chunking_client import (
    LivepeerRemoteFallbackTransportClient,
    LivepeerRemoteHttpChunkingClient,
    fallback_offering_for_streaming,
)


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


class FakeChunkHttpClient:
    def __init__(self):
        self.calls = []
        self.chunk_texts = ["hello one", "hello two", "hello three"]

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/v1/jobs"):
            return FakeResponse(
                {
                    "job_id": f"job-{len(self.calls)}",
                    "work_id": f"work-{len(self.calls)}",
                    "broker_url": "https://broker.example",
                    "mode": "http-multipart@v0",
                    "payment_envelope": "pay",
                    "settle_endpoint": f"/v1/jobs/job-{len(self.calls)}/settle",
                }
            )
        if url.endswith("/v1/cap"):
            text = self.chunk_texts.pop(0)
            return FakeResponse({"text": text}, headers={"X-Livepeer-Work-Units": "1"})
        if url.endswith("/settle"):
            return FakeResponse({"ok": True})
        raise AssertionError(url)


class FakeSilenceThenSpeechHttpClient(FakeChunkHttpClient):
    def __init__(self):
        super().__init__()
        self.cap_calls = 0

    def post(self, url, **kwargs):
        if url.endswith("/v1/cap"):
            self.calls.append((url, kwargs))
            self.cap_calls += 1
            if self.cap_calls == 1:
                return FakeResponse(
                    {"error": {"type": "server_error", "message": "uploaded chunk is silence"}},
                    status_code=500,
                )
            text = self.chunk_texts.pop(0)
            return FakeResponse({"text": text}, headers={"X-Livepeer-Work-Units": "1"})
        return super().post(url, **kwargs)


class FakeBrokerFailureHttpClient(FakeChunkHttpClient):
    def post(self, url, **kwargs):
        if url.endswith("/v1/cap"):
            self.calls.append((url, kwargs))
            return FakeResponse(
                {"error": {"type": "server_error", "message": "GPU out of memory"}},
                status_code=500,
            )
        return super().post(url, **kwargs)


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
                    "billed_value_wei": 0,
                    "refund_wei": 20,
                    "outcome": kwargs["json"].get("outcome", ""),
                    "closed_at": "2026-06-11T17:01:00Z",
                }
            )
        raise AssertionError(url)


def test_fallback_offering_maps_stream_variant_to_http_variant():
    assert fallback_offering_for_streaming("nemo-meeting-stream") == "nemo-meeting"
    assert fallback_offering_for_streaming("nemo-meeting") == "nemo-meeting"


def test_http_chunking_session_batches_pcm_frames_and_finishes():
    http_client = FakeChunkHttpClient()
    client = LivepeerRemoteHttpChunkingClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        offering="nemo-meeting",
        chunk_size_seconds=0.16,
        http_client=http_client,
    )

    with client.connect_session(session_id="stream_1") as session:
        assert session.send_audio_frame(b"\x00\x00" * 1280) == []
        emitted = session.send_audio_frame(b"\x00\x00" * 1280, source_event={"frame_index": 1})
        finished = session.finish()

    assert [event["event_type"] for event in emitted] == [
        "transcript.segment",
        "payment.job.completed",
    ]
    assert emitted[0]["text"] == "hello one"
    assert emitted[0]["source_event"] == {"frame_index": 1}
    assert finished[-1]["event_type"] == "transcript.session.finished"
    assert any("/v1/jobs" in call[0] for call in http_client.calls)
    assert any(call[0].endswith("/v1/cap") for call in http_client.calls)


def test_http_chunking_session_skips_silence_only_broker_failure_and_continues():
    http_client = FakeSilenceThenSpeechHttpClient()
    client = LivepeerRemoteHttpChunkingClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        offering="nemo-meeting",
        chunk_size_seconds=0.16,
        http_client=http_client,
    )

    with client.connect_session(session_id="stream_1") as session:
        assert session.send_audio_frame(b"\x00\x00" * 1280) == []
        skipped = session.send_audio_frame(b"\x00\x00" * 1280, source_event={"frame_index": 1})
        assert session.send_audio_frame(b"\x01\x00" * 1280) == []
        emitted = session.send_audio_frame(b"\x01\x00" * 1280, source_event={"frame_index": 3})
        finished = session.finish()

    assert [event["event_type"] for event in skipped] == ["transcript.chunk.skipped"]
    assert skipped[0]["skip_reason"] == "silence"
    assert skipped[0]["text_status"] == "silence"
    assert skipped[0]["actual_units"] == 0
    assert skipped[0]["source_event"] == {"frame_index": 1}
    assert "uploaded chunk is silence" in skipped[0]["error"]
    assert [event["event_type"] for event in emitted] == [
        "transcript.segment",
        "payment.job.completed",
    ]
    assert emitted[0]["text"] == "hello one"
    assert emitted[0]["start"] == skipped[0]["end"]
    assert finished[-1]["event_type"] == "transcript.session.finished"
    settle_calls = [call for call in http_client.calls if call[0].endswith("/settle")]
    assert settle_calls[0][1]["json"]["actual_units"] == 0


def test_http_chunking_session_fast_fails_non_silence_broker_errors():
    http_client = FakeBrokerFailureHttpClient()
    client = LivepeerRemoteHttpChunkingClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        offering="nemo-meeting",
        chunk_size_seconds=0.16,
        http_client=http_client,
    )

    with pytest.raises(RuntimeError, match="GPU out of memory"):
        with client.connect_session(session_id="stream_1") as session:
            session.send_audio_frame(b"\x00\x00" * 1280)
            session.send_audio_frame(b"\x00\x00" * 1280)


def test_fallback_transport_switches_to_http_chunking_when_ws_handshake_fails():
    realtime_http_client = FakeRealtimeHttpClient()
    chunk_http_client = FakeChunkHttpClient()

    def failing_connect(url, **kwargs):
        raise RuntimeError("EOF before HTTP response")

    client = LivepeerRemoteFallbackTransportClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        capability="openai:audio-transcriptions",
        realtime_offering="nemo-meeting-stream",
        estimated_runway_units=60,
        max_total_units=90,
        chunk_size_seconds=0.16,
        websocket_connect=failing_connect,
        realtime_http_client=realtime_http_client,
        chunk_http_client=chunk_http_client,
    )

    with client.connect_session(session_id="requested-session") as session:
        initial_events = list(session.events)
        session.send_audio_frame(b"\x00\x00" * 1280)
        emitted = session.send_audio_frame(b"\x00\x00" * 1280)
        finished = session.finish()

    assert any(event["event_type"] == "livepeer.realtime.websocket.handshake.failed" for event in initial_events)
    assert any(event["event_type"] == "payment.session.closed" for event in initial_events)
    assert any(event["event_type"] == "transcript.transport.fallback" for event in initial_events)
    assert emitted[0]["event_type"] == "transcript.segment"
    assert emitted[0]["transport"] == "livepeer_remote_http_chunking"
    assert finished[-1]["event_type"] == "transcript.session.finished"
    close_call = realtime_http_client.calls[-1]
    assert close_call[0] == "post"
    assert close_call[1].endswith("/v1/sessions/loc-session-1/close")
    assert close_call[2]["json"]["outcome"] == "ws_handshake_failed_fallback_to_http_chunking"
