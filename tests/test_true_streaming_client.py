import wave
import os

import pytest

from roboflow_livepeer_blocks.true_streaming import (
    LivepeerTrueStreamingSessionConfig,
    LivepeerVDONinjaDirectTrueStreamingRunner,
    NemoTrueStreamingWebSocketClient,
    _true_streaming_ws_url,
    build_local_audio_ingest_true_streaming_runner,
    build_vdo_direct_true_streaming_runner,
    iter_pcm16_wav_frames,
)


def test_true_streaming_ws_url_maps_http_runner_to_stream_endpoint():
    assert _true_streaming_ws_url(
        "http://runner:8080",
        session_id="stream_1",
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
    ) == (
        "ws://runner:8080/v1/audio/transcriptions/stream?"
        "session_id=stream_1&language=en&preset=meeting&max_speakers=4&sample_rate=16000"
    )


def test_iter_pcm16_wav_frames_reads_fixed_duration_frames(tmp_path):
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    frames = list(
        iter_pcm16_wav_frames(
            audio_path,
            sample_rate=16000,
            frame_duration_seconds=0.05,
        )
    )

    assert [len(frame) for frame in frames] == [1600, 1600]


def test_nemo_ws_session_does_not_require_initial_event_before_audio():
    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.recv_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def send(self, payload):
            self.sent.append(payload)

        def recv(self, timeout=None):
            self.recv_calls += 1
            if self.sent and isinstance(self.sent[-1], bytes) and self.recv_calls == 1:
                return '{"event_type":"transcript.segment","text":"hello"}'
            raise TimeoutError(f"timed out after {timeout}")

    fake_ws = FakeWebSocket()
    client = NemoTrueStreamingWebSocketClient(
        base_url="http://runner:8080",
        websocket_connect=lambda url: fake_ws,
        initial_receive_timeout_seconds=90,
    )

    with client.connect_session(session_id="stream_1") as session:
        assert fake_ws.recv_calls == 0
        emitted = session.send_audio_frame(b"\x00\x00" * 1280)

    assert emitted == [{"event_type": "transcript.segment", "text": "hello"}]


def test_nemo_ws_session_finish_tolerates_runner_closed_transport():
    class FakeConnectionClosedError(Exception):
        pass

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def send(self, payload):
            if isinstance(payload, str):
                raise FakeConnectionClosedError("no close frame received or sent")
            self.sent.append(payload)

        def recv(self, timeout=None):
            raise TimeoutError(f"timed out after {timeout}")

    fake_ws = FakeWebSocket()
    client = NemoTrueStreamingWebSocketClient(
        base_url="http://runner:8080",
        websocket_connect=lambda url: fake_ws,
    )

    with client.connect_session(session_id="stream_1") as session:
        session.send_audio_frame(b"\x00\x00" * 1280)
        finished = session.finish()

    assert finished == [
        {
            "event_type": "transcript.session.finished",
            "session_id": "",
            "status": "closed",
            "finish_reason": "websocket_closed_before_finish_ack",
            "transport_error_type": "FakeConnectionClosedError",
            "transport_error": "no close frame received or sent",
        }
    ]


def test_nemo_ws_session_finish_marks_closed_when_finish_ack_times_out():
    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def send(self, payload):
            self.sent.append(payload)

        def recv(self, timeout=None):
            raise TimeoutError(f"timed out after {timeout}")

    fake_ws = FakeWebSocket()
    client = NemoTrueStreamingWebSocketClient(
        base_url="http://runner:8080",
        websocket_connect=lambda url: fake_ws,
        finish_receive_timeout_seconds=0.01,
    )

    with client.connect_session(session_id="stream_1") as session:
        session.send_audio_frame(b"\x00\x00" * 1280)
        finished = session.finish()

    assert finished == [
        {
            "event_type": "transcript.session.finished",
            "session_id": "",
            "status": "closed",
            "finish_reason": "finish_ack_timeout",
        }
    ]


def test_vdo_direct_runner_zero_duration_pumps_until_pipe_closes(tmp_path):
    class FakeStdout:
        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

    class FakeProcess:
        def __init__(self, fd=None):
            self.stdout = FakeStdout(fd) if fd is not None else None
            self.returncode = 0

        def poll(self):
            return None

    class FakeSession:
        def __init__(self):
            self.frames = []

        def send_audio_frame(self, frame, *, source_event=None):
            self.frames.append((frame, source_event))
            return []

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"\x01\x02" * 1280 * 2)
    os.close(write_fd)
    session = FakeSession()
    runner = LivepeerVDONinjaDirectTrueStreamingRunner(
        source="stream_123",
        client=None,
        session_config=LivepeerTrueStreamingSessionConfig(
            session_id="stream_test",
            sample_rate=16000,
            frame_duration_seconds=0.08,
        ),
        artifact_dir=tmp_path,
        duration_seconds=0,
        startup_timeout_seconds=1,
        password="",
        signaling_server="",
        buffer_ms=300,
    )

    sent_frame_count, sent_audio_bytes = runner._pump_resampled_pcm(
        session=session,
        events=[],
        ffmpeg=FakeProcess(read_fd),
        publisher=FakeProcess(),
        frame_bytes=2560,
    )
    os.close(read_fd)

    assert sent_frame_count == 2
    assert sent_audio_bytes == 5120
    assert len(session.frames) == 2


def test_vdo_direct_runner_audio_startup_timeout_reports_source_context(
    monkeypatch, tmp_path
):
    class FakeStdout:
        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

    class FakeProcess:
        def __init__(self, fd=None):
            self.stdout = FakeStdout(fd) if fd is not None else None
            self.returncode = None

        def poll(self):
            return None

    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    runner = LivepeerVDONinjaDirectTrueStreamingRunner(
        source="stream_123",
        client=None,
        session_config=LivepeerTrueStreamingSessionConfig(
            session_id="stream_test",
            sample_rate=16000,
            frame_duration_seconds=0.08,
        ),
        artifact_dir=tmp_path,
        duration_seconds=30,
        startup_timeout_seconds=0.5,
        password="",
        signaling_server="wss://bridge.test:9443",
        buffer_ms=300,
    )
    runner.publisher_log_path.write_text("NO HEARTBEAT\n", encoding="utf-8")
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.true_streaming.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.true_streaming.select.select",
        lambda *args, **kwargs: ([], [], []),
    )

    with pytest.raises(RuntimeError) as exc_info:
        runner._pump_resampled_pcm(
            session=object(),
            events=[],
            ffmpeg=FakeProcess(read_fd),
            publisher=FakeProcess(),
            frame_bytes=2560,
        )
    os.close(read_fd)

    message = str(exc_info.value)
    assert "timed out after 0.5s waiting for live VDO audio" in message
    assert "source='stream_123'" in message
    assert "stream_id='stream_123'" in message
    assert "signaling_server='wss://bridge.test:9443'" in message
    assert "publisher_log_path=" in message
    assert "NO HEARTBEAT" in message


def test_vdo_direct_runner_audio_idle_timeout_after_partial_pcm(monkeypatch, tmp_path):
    class FakeStdout:
        def __init__(self, fd):
            self._fd = fd

        def fileno(self):
            return self._fd

    class FakeProcess:
        def __init__(self, fd=None):
            self.stdout = FakeStdout(fd) if fd is not None else None
            self.returncode = None

        def poll(self):
            return None

    class FakeSession:
        def __init__(self):
            self.frames = []

        def send_audio_frame(self, frame, *, source_event=None):
            self.frames.append((frame, source_event))
            return []

    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"\x01\x02" * 1280)
    runner = LivepeerVDONinjaDirectTrueStreamingRunner(
        source="stream_123",
        client=None,
        session_config=LivepeerTrueStreamingSessionConfig(
            session_id="stream_test",
            sample_rate=16000,
            frame_duration_seconds=0.08,
        ),
        artifact_dir=tmp_path,
        duration_seconds=30,
        startup_timeout_seconds=0.5,
        password="",
        signaling_server="wss://bridge.test:9443",
        buffer_ms=300,
    )
    monotonic_values = iter([0.0, 0.1, 0.2, 0.7])
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.true_streaming.time.monotonic",
        lambda: next(monotonic_values),
    )
    select_calls = {"count": 0}

    def fake_select(readers, *_args):
        select_calls["count"] += 1
        if select_calls["count"] == 1:
            return readers, [], []
        return [], [], []

    monkeypatch.setattr("roboflow_livepeer_blocks.true_streaming.select.select", fake_select)
    events = []
    try:
        sent_frame_count, sent_audio_bytes = runner._pump_resampled_pcm(
            session=FakeSession(),
            events=events,
            ffmpeg=FakeProcess(read_fd),
            publisher=FakeProcess(),
            frame_bytes=2560,
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert sent_frame_count == 1
    assert sent_audio_bytes == 2560
    assert events[-1]["event_type"] == "source.audio_idle_timeout"
    assert events[-1]["sent_audio_seconds"] == 0.08


def test_vdo_direct_builder_passes_startup_timeout_to_ws_client(monkeypatch, tmp_path):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.true_streaming.resolve_vdo_stream_source",
        lambda **kwargs: {
            "source": "stream_123",
            "stream_id": "stream_123",
            "signaling_server": "wss://bridge:9443",
            "password": "false",
        },
    )

    runner = build_vdo_direct_true_streaming_runner(
        source="auto",
        runner_url="http://runner:8080",
        output_dir=str(tmp_path),
        duration_seconds=30,
        startup_timeout_seconds=75,
        session_id="",
        password="",
        signaling_server="wss://bridge:9443",
        buffer_ms=300,
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
        frame_duration_seconds=0.08,
        client_cls=FakeClient,
    )

    assert runner.client.__class__ is FakeClient
    assert runner.password == "false"
    assert captured["base_url"] == "http://runner:8080"
    assert captured["initial_receive_timeout_seconds"] == 75


def test_local_audio_builder_passes_startup_timeout_to_ws_client(tmp_path):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeIngestClient:
        def __init__(self, **kwargs):
            pass

    runner = build_local_audio_ingest_true_streaming_runner(
        source="ws://127.0.0.1:8876/v1/ingest/audio/test/consume",
        runner_url="http://runner:8080",
        local_audio_ingest_url="http://local-audio-ingest:8876",
        output_dir=str(tmp_path),
        duration_seconds=30,
        startup_timeout_seconds=90,
        session_id="",
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
        frame_duration_seconds=0.08,
        client_cls=FakeClient,
        ingest_client_cls=FakeIngestClient,
    )

    assert runner.client.__class__ is FakeClient
    assert captured["base_url"] == "http://runner:8080"
    assert captured["initial_receive_timeout_seconds"] == 90


def test_vdo_direct_builder_does_not_pass_ws_timeout_to_http_only_client(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.true_streaming.resolve_vdo_stream_source",
        lambda **kwargs: {
            "source": "stream_123",
            "stream_id": "stream_123",
            "signaling_server": "wss://bridge:9443",
            "password": "false",
        },
    )
    captured = {}

    class FakeHttpClient:
        def __init__(self, *, base_url, chunk_size_seconds):
            captured["base_url"] = base_url
            captured["chunk_size_seconds"] = chunk_size_seconds

    runner = build_vdo_direct_true_streaming_runner(
        source="stream_123",
        runner_url="http://runner:8080",
        output_dir=str(tmp_path),
        duration_seconds=30,
        startup_timeout_seconds=75,
        session_id="",
        password="",
        signaling_server="wss://bridge:9443",
        buffer_ms=300,
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
        frame_duration_seconds=0.08,
        client_cls=FakeHttpClient,
        client_init_kwargs={"chunk_size_seconds": 30.08},
    )

    assert runner.client.__class__ is FakeHttpClient
    assert captured == {
        "base_url": "http://runner:8080",
        "chunk_size_seconds": 30.08,
    }


def test_local_audio_builder_does_not_pass_ws_timeout_to_http_only_client(tmp_path):
    captured = {}

    class FakeHttpClient:
        def __init__(self, *, base_url, chunk_size_seconds):
            captured["base_url"] = base_url
            captured["chunk_size_seconds"] = chunk_size_seconds

    class FakeIngestClient:
        def __init__(self, **kwargs):
            pass

    runner = build_local_audio_ingest_true_streaming_runner(
        source="ws://127.0.0.1:8876/v1/ingest/audio/test/consume",
        runner_url="http://runner:8080",
        local_audio_ingest_url="http://local-audio-ingest:8876",
        output_dir=str(tmp_path),
        duration_seconds=30,
        startup_timeout_seconds=90,
        session_id="",
        language="en",
        preset="meeting",
        max_speakers=4,
        sample_rate=16000,
        frame_duration_seconds=0.08,
        client_cls=FakeHttpClient,
        ingest_client_cls=FakeIngestClient,
        client_init_kwargs={"chunk_size_seconds": 30.08},
    )

    assert runner.client.__class__ is FakeHttpClient
    assert captured == {
        "base_url": "http://runner:8080",
        "chunk_size_seconds": 30.08,
    }
