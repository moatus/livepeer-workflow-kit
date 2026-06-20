from pathlib import Path

from roboflow_livepeer_blocks.block import (
    LivepeerAudioDiarizedTranscribeLocalV1,
    LivepeerAudioTranscribeV1,
    LivepeerFlorence2ScreenSlideAnalysisV1,
    LivepeerLocalAudioIngressLiveAudioSourceV1,
    LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1,
    LivepeerPCM16AudioTransformV1,
    LivepeerScreenSlideCaptureV1,
    LivepeerTranscriptOutputV1,
    LivepeerTrueStreamingTranscriptionSessionV1,
    LivepeerVDONinjaDirectTrueStreamingSessionV1,
    LivepeerVDONinjaLiveDiarizedSessionV1,
    LivepeerVDONinjaLiveAudioSourceV1,
    LivepeerVDONinjaMediaSourceV1,
    LivepeerVDONinjaRollingAudioCaptureV1,
    LivepeerVDONinjaTrueStreamingSessionV1,
)
from roboflow_livepeer_blocks.livepeer_http_chunking_client import LivepeerRemoteHttpChunkingClient
from roboflow_livepeer_blocks.livepeer_http_chunking_client import (
    LivepeerRemoteHttpChunkingClient,
)


class FakeClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def transcribe_audio_file(self, **kwargs):
        self.kwargs = kwargs
        return {
            "text": "hello world",
            "chunks": [{"index": 0}],
            "actual_units": 12,
            "job_ids": ["job-1"],
            "work_ids": ["work-1"],
            "raw_responses": [{"ok": True}],
        }


class FakeDiarizedClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def diarized_transcribe_audio_file(self, **kwargs):
        self.kwargs = kwargs
        return {
            "text": "speaker_0: hello",
            "speaker_count": 1,
            "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
            "segments": [{"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": "hello"}],
            "words": [{"speaker": "speaker_0", "start": 0.0, "end": 0.5, "word": "hello"}],
            "artifacts": {"json_path": "/tmp/job/audio.json"},
            "usage": {"work_units": 2},
            "actual_units": 2,
            "api_endpoint": "/v1/audio/transcriptions",
        }

    def create_live_session(self, **kwargs):
        self.create_kwargs = kwargs
        return {
            "event_type": "session.started",
            "session_id": kwargs.get("session_id") or "live_fake",
            "status": "active",
        }

    def ingest_live_audio_file(self, **kwargs):
        self.ingest_kwargs = kwargs
        return {
            "event_type": "audio.ingested",
            "session_id": kwargs["session_id"],
            "status": "active",
            "chunk": {
                "sequence_index": kwargs["sequence_index"],
                "start": 0.0,
                "end": 1.0,
                "duration_seconds": 1.0,
                "text": "hello live",
                "text_status": "available",
                "asr_model": "fake-live-asr",
            },
            "segments": [{"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": "hello live"}],
            "new_segments": [
                {"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": "hello live"}
            ],
        }

    def finish_live_session(self, **kwargs):
        self.finish_kwargs = kwargs
        return {
            "event_type": "session.finished",
            "session_id": kwargs["session_id"],
            "status": "closed",
            "final_audio_path": "/tmp/audio-diarized-transcription-runner/live_sessions/live_fake/session.wav",
            "segments": [{"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": ""}],
            "final_transcription": {
                "text": "speaker_0: hello",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
                "segments": [{"speaker": "speaker_0", "start": 0.0, "end": 1.0, "text": "hello"}],
                "words": [{"speaker": "speaker_0", "start": 0.0, "end": 0.5, "word": "hello"}],
            },
        }


class FakeLiveSegment:
    stream_id = "stream_123"
    index = 0
    audio_path = "/tmp/seg0.wav"

    def as_dict(self):
        return {
            "stream_id": self.stream_id,
            "index": self.index,
            "audio_path": self.audio_path,
            "audio_duration_seconds": 1.0,
        }


class FakeLiveSource:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stream_id = "stream_123"
        self.closed = False

    def open(self):
        self.opened = True

    def segments(self):
        yield FakeLiveSegment()

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeTrueStreamingSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.events = [
            {
                "event_type": "session.snapshot",
                "session_id": session_id,
                "status": "active",
                "models": {
                    "streaming_asr": "nvidia/nemotron-speech-streaming-en-0.6b",
                    "streaming_diarization": "nvidia/diar_streaming_sortformer_4spk-v2.1",
                },
            }
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def send_audio_file(self, audio_path, source_segment=None):
        return [
            {
                "event_type": "speaker.update",
                "session_id": self.session_id,
                "start": 0.0,
                "end": 1.0,
                "speaker": "speaker_0",
                "is_provisional": True,
                "is_final": False,
                "source_segment": source_segment,
            },
            {
                "event_type": "transcript.segment",
                "session_id": self.session_id,
                "start": 0.0,
                "end": 1.0,
                "speaker": "speaker_0",
                "text": "hello true stream",
                "is_provisional": True,
                "is_final": False,
                "source_segment": source_segment,
            },
        ]

    def finish(self):
        return [
            {
                "event_type": "transcript.segment",
                "session_id": self.session_id,
                "start": 0.0,
                "end": 1.0,
                "speaker": "speaker_0",
                "text": "hello true stream",
                "is_provisional": False,
                "is_final": True,
            },
            {
                "event_type": "transcript.session.finished",
                "session_id": self.session_id,
                "status": "closed",
                "is_final": True,
            },
        ]


class FakeTrueStreamingClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def connect_session(self, **kwargs):
        self.kwargs = kwargs
        return FakeTrueStreamingSession(kwargs["session_id"])


class FakeFlorenceAnalyzer:
    def __init__(self, *, model_id, api_key=None, api_url=""):
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url

    def analyze_image(self, image_path):
        if image_path.endswith("frame_000001.jpg"):
            return {
                "caption": "A presentation slide is visible",
                "detailed_caption": "A slide with a title and several bullet points.",
                "ocr_text": "Q3 roadmap\nHiring plan\nBudget update",
            }
        return {
            "caption": "The same presentation slide remains on screen",
            "detailed_caption": "The same slide remains visible with unchanged bullets and meeting chat on the side.",
            "ocr_text": "Q3 roadmap\nHiring plan\nBudget update\nChat\nTo everyone: looks good\nMute\nLeave",
        }


def test_block_returns_named_aggregate_fields():
    block = LivepeerAudioTranscribeV1(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        client_cls=FakeClient,
    )

    result = block.run(audio_path="/tmp/audio.mp3")

    assert result["text"] == "hello world"
    assert result["actual_units"] == 12
    assert result["job_ids"] == ["job-1"]
    assert result["result"]["work_ids"] == ["work-1"]


def test_local_diarized_block_returns_normalized_fields():
    block = LivepeerAudioDiarizedTranscribeLocalV1(
        runner_url="http://runner:8080",
        client_cls=FakeDiarizedClient,
    )

    result = block.run(audio_path="/tmp/audio.wav", num_speakers=1, max_speakers=4)

    assert result["text"] == "speaker_0: hello"
    assert result["speaker_count"] == 1
    assert result["actual_units"] == 2
    assert result["api_endpoint"] == "/v1/audio/transcriptions"
    assert result["segments"][0]["speaker"] == "speaker_0"
    assert result["result"]["artifacts"]["json_path"] == "/tmp/job/audio.json"


def test_vdo_rolling_audio_capture_block_returns_segment_fields(monkeypatch):
    def fake_capture_rolling_audio_segments(**kwargs):
        assert kwargs["source"] == "stream_123"
        assert kwargs["segment_count"] == 2
        return {
            "stream_id": "stream_123",
            "segments": [
                {"index": 0, "audio_path": "/tmp/seg0.wav"},
                {"index": 1, "audio_path": "/tmp/seg1.wav"},
            ],
            "audio_paths": ["/tmp/seg0.wav", "/tmp/seg1.wav"],
            "first_audio_path": "/tmp/seg0.wav",
            "latest_audio_path": "/tmp/seg1.wav",
            "segment_count": 2,
            "output_dir": "/tmp",
        }

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.capture_rolling_audio_segments",
        fake_capture_rolling_audio_segments,
    )

    block = LivepeerVDONinjaRollingAudioCaptureV1()
    result = block.run(source="stream_123", segment_count=2)

    assert result["stream_id"] == "stream_123"
    assert result["latest_audio_path"] == "/tmp/seg1.wav"
    assert result["audio_paths"] == ["/tmp/seg0.wav", "/tmp/seg1.wav"]
    assert result["result"]["segment_count"] == 2


def test_vdo_live_diarized_session_block_persists_events_and_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.LivepeerVDONinjaAudioSegmentSource",
        FakeLiveSource,
    )
    block = LivepeerVDONinjaLiveDiarizedSessionV1(
        runner_url="http://runner:8080",
        client_cls=FakeDiarizedClient,
    )

    result = block.run(
        source="stream_123",
        segment_count=1,
        output_dir=str(tmp_path),
        session_id="live_fake",
    )

    assert result["session_id"] == "live_fake"
    assert result["status"] == "closed"
    assert result["text"] == "speaker_0: hello"
    assert result["audio_paths"] == ["/tmp/seg0.wav"]
    assert result["live_segments"][0]["speaker"] == "speaker_0"
    assert result["result"]["event_count"] == 4
    assert result["result"]["created"]["event_type"] == "session.started"
    assert result["result"]["finished"]["event_type"] == "session.finished"
    assert result["events_jsonl_path"]
    assert result["provisional_transcript_jsonl_path"]
    assert result["transcript_event_count"] >= 3
    assert result["transcript_events"][0]["event_type"] == "transcript.session.started"
    assert result["transcript_events"][1]["event_type"] == "transcript.chunk_ingested"
    assert result["transcript_events"][1]["text"] == "hello live"
    assert result["transcript_events"][2]["text_status"] == "available"
    assert result["transcript_events"][2]["text"] == "hello live"
    assert result["result"]["provisional_transcript_jsonl_path"] == result["provisional_transcript_jsonl_path"]
    assert result["result_json_path"]
    assert result["transcript_text_path"]


def test_vdo_true_streaming_session_block_uses_persistent_session_runner(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.true_streaming.LivepeerVDONinjaAudioSegmentSource",
        FakeLiveSource,
    )
    block = LivepeerVDONinjaTrueStreamingSessionV1(
        runner_url="http://runner:8080",
        client_cls=FakeTrueStreamingClient,
    )

    result = block.run(
        source="stream_123",
        segment_count=1,
        output_dir=str(tmp_path),
        session_id="stream_fake",
    )

    assert result["session_id"] == "stream_fake"
    assert result["status"] == "closed"
    assert result["text"] == "hello true stream"
    assert result["audio_paths"] == ["/tmp/seg0.wav"]
    assert result["speaker_count"] == 1
    assert result["speakers"][0]["id"] == "speaker_0"
    assert result["transcript_event_count"] == 3
    assert result["events_jsonl_path"]
    assert result["result_json_path"]
    assert result["transcript_text_path"]
    assert result["result"]["events"][0]["event_type"] == "session.snapshot"
    assert result["result"]["events"][1]["event_type"] == "source.audio_chunk"


def test_vdo_live_audio_source_block_returns_descriptor():
    block = LivepeerVDONinjaLiveAudioSourceV1()

    result = block.run(
        source="https://vdo.ninja/?view=stream_123&bitrate=6000&codec=h264",
        signaling_server="wss://localhost:9443",
        buffer_ms=250,
    )

    descriptor = result["source_descriptor"]
    assert descriptor["kind"] == "live_audio_source"
    assert descriptor["source_type"] == "vdo_ninja"
    assert descriptor["stream_id"] == "stream_123"
    assert descriptor["publisher"] == "raspberry_ninja_fdsink"
    assert descriptor["sample_format"] == "s16le"
    assert descriptor["sample_rate"] == 48000
    assert descriptor["channels"] == 1
    assert descriptor["buffer_ms"] == 250
    assert descriptor["signaling_server"] == "wss://localhost:9443"
    assert descriptor["available_tracks"] == ["audio", "video"]
    assert descriptor["selected_tracks"] == ["audio"]
    assert descriptor["media_source"]["kind"] == "live_media_source_ref"


def test_vdo_live_audio_source_block_resolves_auto_source(monkeypatch):
    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.resolve_vdo_stream_source",
        lambda **kwargs: {
            "source": "stream_new",
            "stream_id": "stream_new",
            "signaling_server": kwargs["signaling_server"],
            "password": "false",
            "auto_resolved": True,
            "requested_source": kwargs["source"],
            "status_url": "https://localhost:9443/statusz",
        },
    )
    block = LivepeerVDONinjaLiveAudioSourceV1()

    result = block.run(source="auto", signaling_server="wss://localhost:9443")

    descriptor = result["source_descriptor"]
    assert result["source"] == "stream_new"
    assert result["stream_id"] == "stream_new"
    assert descriptor["source"] == "stream_new"
    assert descriptor["stream_id"] == "stream_new"
    assert descriptor["auto_resolved"] is True
    assert descriptor["requested_source"] == "auto"
    assert descriptor["bridge_status_url"] == "https://localhost:9443/statusz"
    assert descriptor["signaling_server"] == "wss://localhost:9443"
    assert descriptor["password"] == "false"


def test_vdo_media_source_block_returns_audio_and_video_descriptors():
    block = LivepeerVDONinjaMediaSourceV1()

    result = block.run(
        source="https://vdo.ninja/?view=stream_123&bitrate=6000&codec=h264",
        signaling_server="wss://localhost:9443",
        buffer_ms=250,
        video_frame_rate=0.5,
    )

    media = result["media_descriptor"]
    audio = result["audio_source_descriptor"]
    video = result["video_source_descriptor"]
    assert result["tracks"] == ["audio", "video"]
    assert media["kind"] == "live_media_source"
    assert media["stream_id"] == "stream_123"
    assert media["audio_source_descriptor"] == audio
    assert media["video_source_descriptor"] == video
    assert audio["kind"] == "live_audio_source"
    assert audio["selected_tracks"] == ["audio"]
    assert audio["media_source"]["kind"] == "live_media_source_ref"
    assert video["kind"] == "live_video_source"
    assert video["selected_tracks"] == ["video"]
    assert video["consumer"] == "future_frame_sampler"
    assert video["default_frame_rate"] == 0.5
    assert video["signaling_server"] == "wss://localhost:9443"


def test_vdo_media_source_block_accepts_bridge_url_as_source(monkeypatch):
    captured = {}

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return {
            "source": "stream_new",
            "stream_id": "stream_new",
            "signaling_server": "wss://localhost:9443",
            "password": "false",
            "auto_resolved": True,
            "requested_source": "auto",
            "status_url": "https://localhost:9443/statusz",
        }

    monkeypatch.setattr("roboflow_livepeer_blocks.block.resolve_vdo_stream_source", fake_resolve)
    block = LivepeerVDONinjaMediaSourceV1()

    result = block.run(source="wss://localhost:9443")

    assert captured["source"] == "wss://localhost:9443"
    assert result["stream_id"] == "stream_new"
    assert result["media_descriptor"]["auto_resolved"] is True
    assert result["audio_source_descriptor"]["source"] == "stream_new"
    assert result["video_source_descriptor"]["source"] == "stream_new"
    assert result["audio_source_descriptor"]["password"] == "false"
    assert result["video_source_descriptor"]["password"] == "false"


def test_pcm16_audio_transform_block_returns_descriptor():
    source_descriptor = {
        "kind": "live_audio_source",
        "source_type": "vdo_ninja",
        "source": "stream_123",
        "stream_id": "stream_123",
        "sample_format": "s16le",
        "sample_rate": 48000,
        "channels": 1,
    }
    block = LivepeerPCM16AudioTransformV1()

    result = block.run(
        source_descriptor=source_descriptor,
        sample_rate=16000,
        frame_duration_seconds=0.08,
    )

    descriptor = result["pcm_descriptor"]
    assert descriptor["kind"] == "pcm16_audio_stream"
    assert descriptor["source_descriptor"] == source_descriptor
    assert descriptor["input_sample_rate"] == 48000
    assert descriptor["sample_rate"] == 16000
    assert descriptor["channels"] == 1
    assert descriptor["resampler"] == "ffmpeg"


def test_local_audio_ingest_source_block_returns_descriptor():
    block = LivepeerLocalAudioIngressLiveAudioSourceV1(
        local_audio_ingest_url="http://local-audio-ingest:8765"
    )

    result = block.run(source="ws://127.0.0.1:8765/v1/ingest/audio/test-session")

    descriptor = result["source_descriptor"]
    assert descriptor["kind"] == "live_audio_source"
    assert descriptor["source_type"] == "localhost_ingest"
    assert descriptor["stream_id"] == "test-session"
    assert descriptor["publisher"] == "localhost_ingest_ws"
    assert descriptor["sample_format"] == "s16le"
    assert descriptor["sample_rate"] == 16000
    assert descriptor["channels"] == 1
    assert descriptor["consume_url"] == "ws://127.0.0.1:8765/v1/ingest/audio/test-session/consume"


def test_true_streaming_transcription_session_block_runs_pcm_descriptor(monkeypatch):
    class FakeComposableRunner:
        def run(self):
            return {
                "session_id": "stream_composable",
                "stream_id": "stream_123",
                "status": "closed",
                "source_mode": "vdo_ninja_fdsink_live_pcm",
                "source": "stream_123",
                "sent_audio_seconds": 30.0,
                "sent_frame_count": 375,
                "text": "hello composable stream",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "source.connected"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
                "publisher_log_path": "/tmp/raspberry.log",
                "ffmpeg_log_path": "/tmp/ffmpeg.log",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeComposableRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_vdo_direct_true_streaming_runner",
        fake_build_runner,
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "vdo_ninja",
            "source": "stream_123",
            "stream_id": "stream_123",
            "password": "",
            "signaling_server": "wss://vdo-signaling-bridge:9443",
            "buffer_ms": 300,
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }
    block = LivepeerTrueStreamingTranscriptionSessionV1(runner_url="http://runner:8080")

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=30,
        session_id="stream_composable",
    )

    assert captured["runner_url"] == "http://runner:8080"
    assert captured["source"] == "stream_123"
    assert captured["signaling_server"] == "wss://vdo-signaling-bridge:9443"
    assert captured["duration_seconds"] == 30.0
    assert captured["sample_rate"] == 16000
    assert result["transcription_session"]["pcm_descriptor"] == pcm_descriptor
    assert result["source_mode"] == "vdo_ninja_fdsink_live_pcm"
    assert result["text"] == "hello composable stream"


def test_true_streaming_transcription_session_block_can_use_segmented_vdo_ingest(
    monkeypatch,
):
    class FakeSegmentedRunner:
        def run(self):
            return {
                "session_id": "stream_segmented",
                "stream_id": "stream_123",
                "status": "closed",
                "captured_segments": [
                    {
                        "index": 0,
                        "stream_id": "stream_123",
                        "audio_duration_seconds": 12.0,
                        "requested_duration_seconds": 12.0,
                    },
                    {
                        "index": 1,
                        "stream_id": "stream_123",
                        "audio_duration_seconds": 12.0,
                        "requested_duration_seconds": 12.0,
                    },
                ],
                "audio_paths": ["/tmp/seg0.wav", "/tmp/seg1.wav"],
                "text": "hello segmented stream",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 24.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "source.audio_chunk"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeSegmentedRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_vdo_true_streaming_runner",
        fake_build_runner,
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "vdo_ninja",
            "source": "stream_123",
            "stream_id": "stream_123",
            "password": "false",
            "signaling_server": "wss://vdo-signaling-bridge:9443",
            "buffer_ms": 300,
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }
    block = LivepeerTrueStreamingTranscriptionSessionV1(runner_url="http://runner:8080")

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=24,
        startup_timeout_seconds=45,
        session_id="stream_segmented",
        vdo_ingest_mode="segmented_wav",
        vdo_segment_duration_seconds=12,
        vdo_segment_startup_seconds=6,
    )

    assert captured["runner_url"] == "http://runner:8080"
    assert captured["source"] == "stream_123"
    assert captured["password"] == "false"
    assert captured["signaling_server"] == "wss://vdo-signaling-bridge:9443"
    assert captured["segment_count"] == 2
    assert captured["segment_duration_seconds"] == 12.0
    assert captured["startup_seconds"] == 6.0
    assert captured["audio_only"] is True
    assert captured["client_init_kwargs"]["initial_receive_timeout_seconds"] == 45.0
    assert result["transcription_session"]["pcm_descriptor"] == pcm_descriptor
    assert result["source_mode"] == "vdo_ninja_segmented_wav"
    assert result["sent_audio_seconds"] == 24.0
    assert result["sent_frame_count"] == 300
    assert result["text"] == "hello segmented stream"


def test_true_streaming_transcription_session_block_can_use_explicit_livepeer_remote_http_backend(
    monkeypatch,
):
    class FakeSegmentedRunner:
        def run(self):
            return {
                "session_id": "stream_segmented_http",
                "stream_id": "stream_123",
                "status": "closed",
                "captured_segments": [
                    {
                        "index": 0,
                        "stream_id": "stream_123",
                        "audio_duration_seconds": 12.0,
                        "requested_duration_seconds": 12.0,
                    }
                ],
                "audio_paths": ["/tmp/seg0.wav"],
                "text": "hello segmented http",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 12.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "source.audio_chunk"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeSegmentedRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_vdo_true_streaming_runner",
        fake_build_runner,
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "vdo_ninja",
            "source": "wss://localhost:9443/?view=stream_123&bitrate=6000&codec=h264",
            "stream_id": "stream_123",
            "password": "false",
            "signaling_server": "wss://localhost:9443",
            "buffer_ms": 300,
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }
    block = LivepeerTrueStreamingTranscriptionSessionV1(
        runner_url="http://runner:8080",
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
    )

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=12,
        session_id="stream_segmented_http",
        transcription_backend="livepeer_remote_http",
        vdo_ingest_mode="segmented_wav",
        vdo_segment_duration_seconds=12,
        vdo_segment_startup_seconds=6,
    )

    assert captured["runner_url"] == "https://loc.cloudspe.com"
    assert captured["client_cls"] is LivepeerRemoteHttpChunkingClient
    assert captured["client_init_kwargs"] == {
        "api_key": "test-key",
        "capability": "openai:audio-transcriptions",
        "offering": "nemo-meeting",
        "chunk_size_seconds": 12.08,
    }
    assert captured["source"] == "wss://localhost:9443/?view=stream_123&bitrate=6000&codec=h264"
    assert result["source"] == "wss://localhost:9443/?view=stream_123&bitrate=6000&codec=h264"
    assert result["source_mode"] == "vdo_ninja_segmented_wav"
    assert result["text"] == "hello segmented http"


def test_true_streaming_transcription_session_block_selects_http_only_remote_backend(
    monkeypatch,
):
    class FakeSegmentedRunner:
        def run(self):
            return {
                "session_id": "stream_http",
                "stream_id": "stream_123",
                "status": "closed",
                "captured_segments": [
                    {
                        "index": 0,
                        "stream_id": "stream_123",
                        "audio_duration_seconds": 30.0,
                        "requested_duration_seconds": 30.0,
                    },
                ],
                "audio_paths": ["/tmp/seg0.wav"],
                "text": "hello http",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 30.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "source.audio_chunk"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeSegmentedRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_vdo_true_streaming_runner",
        fake_build_runner,
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "vdo_ninja",
            "source": "https://vdo.ninja/?view=stream_123",
            "stream_id": "stream_123",
            "password": "false",
            "signaling_server": "wss://vdo-signaling-bridge:9443",
            "buffer_ms": 300,
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }
    block = LivepeerTrueStreamingTranscriptionSessionV1(
        runner_url="http://runner:8080",
        api_key="test-key",
        base_url="https://loc.cloudspe.com",
    )

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=30,
        startup_timeout_seconds=45,
        session_id="stream_http",
        transcription_backend="livepeer_remote_http",
        vdo_ingest_mode="segmented_wav",
        vdo_segment_duration_seconds=30,
        livepeer_offering="nemo-meeting-stream",
    )

    assert captured["runner_url"] == "https://loc.cloudspe.com"
    assert captured["source"] == "https://vdo.ninja/?view=stream_123"
    assert captured["client_cls"] is LivepeerRemoteHttpChunkingClient
    assert captured["client_init_kwargs"] == {
        "api_key": "test-key",
        "capability": "openai:audio-transcriptions",
        "offering": "nemo-meeting",
        "chunk_size_seconds": 16.08,
    }
    assert result["vdo_ingest_mode"] == "segmented_wav"
    assert result["text"] == "hello http"


def test_local_audio_ingest_true_streaming_transcription_session_block_runs_pcm_descriptor(
    monkeypatch,
):
    class FakeLocalComposableRunner:
        def run(self):
            return {
                "session_id": "stream_local",
                "stream_id": "test-session",
                "status": "closed",
                "source_mode": "localhost_ingest_pcm",
                "source": "ws://127.0.0.1:8765/v1/ingest/audio/test-session",
                "sent_audio_seconds": 18.0,
                "sent_frame_count": 225,
                "text": "hello localhost stream",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events": [{"event_type": "source.connected"}],
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
                "publisher_log_path": "",
                "ffmpeg_log_path": "",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeLocalComposableRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_local_audio_ingest_true_streaming_runner",
        fake_build_runner,
    )
    pcm_descriptor = {
        "kind": "pcm16_audio_stream",
        "source_descriptor": {
            "kind": "live_audio_source",
            "source_type": "localhost_ingest",
            "source": "ws://127.0.0.1:8765/v1/ingest/audio/test-session",
            "stream_id": "test-session",
            "status_url": "http://127.0.0.1:8765/v1/ingest/audio/test-session",
            "consume_url": "ws://127.0.0.1:8765/v1/ingest/audio/test-session/consume",
        },
        "sample_rate": 16000,
        "frame_duration_seconds": 0.08,
    }
    block = LivepeerLocalAudioIngressTrueStreamingTranscriptionSessionV1(
        runner_url="http://runner:8080",
        local_audio_ingest_url="http://local-audio-ingest:8765",
    )

    result = block.run(
        pcm_descriptor=pcm_descriptor,
        duration_seconds=18,
        session_id="stream_local",
    )

    assert captured["runner_url"] == "http://runner:8080"
    assert captured["local_audio_ingest_url"] == "http://local-audio-ingest:8765"
    assert captured["source"] == "ws://127.0.0.1:8765/v1/ingest/audio/test-session"
    assert captured["duration_seconds"] == 18.0
    assert result["source_mode"] == "localhost_ingest_pcm"
    assert result["text"] == "hello localhost stream"


def test_transcript_output_block_normalizes_session_result():
    session = {
        "session_id": "stream_composable",
        "stream_id": "stream_123",
        "status": "closed",
        "source_mode": "vdo_ninja_fdsink_live_pcm",
        "source": "stream_123",
        "sent_audio_seconds": 30.0,
        "sent_frame_count": 375,
        "text": "hello composable stream",
        "speaker_count": 1,
        "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
        "transcript_events": [{"event_type": "transcript.segment"}],
        "transcript_event_count": 1,
        "events": [
            {"event_type": "source.connected"},
            {"event_type": "audio.frame.received"},
            {"event_type": "audio.frame.received"},
            {"event_type": "transcript.session.finished"},
        ],
        "events_jsonl_path": "/tmp/events.jsonl",
        "result_json_path": "/tmp/result.json",
        "transcript_text_path": "/tmp/transcript.txt",
        "publisher_log_path": "/tmp/raspberry.log",
        "ffmpeg_log_path": "/tmp/ffmpeg.log",
    }
    block = LivepeerTranscriptOutputV1()

    result = block.run(transcription_session=session)

    assert result["text"] == "hello composable stream"
    assert result["speaker_count"] == 1
    assert result["event_counts"]["audio.frame.received"] == 2
    assert result["event_counts"]["source.connected"] == 1
    assert result["result"] == session


def test_screen_slide_capture_block_returns_visual_capture_descriptor():
    video_descriptor = {
        "schema_version": "livepeer.live_video_source.v1",
        "kind": "live_video_source",
        "source_type": "vdo_ninja",
        "source": "stream_123",
        "stream_id": "stream_123",
        "transport": "webrtc",
    }
    block = LivepeerScreenSlideCaptureV1()

    result = block.run(
        video_source_descriptor=video_descriptor,
        duration_seconds=60.0,
        frame_interval_seconds=5.0,
        min_slide_gap_seconds=4.0,
        max_frames=12,
        output_dir="/tmp/visual-capture",
    )

    descriptor = result["capture_descriptor"]
    assert descriptor["kind"] == "screen_slide_capture"
    assert descriptor["schema_version"] == "livepeer.screen_slide_capture.v1"
    assert descriptor["video_source_descriptor"] == video_descriptor
    assert descriptor["duration_seconds"] == 60.0
    assert descriptor["frame_interval_seconds"] == 5.0
    assert descriptor["max_frames"] == 12
    assert descriptor["min_slide_gap_seconds"] == 4.0
    assert descriptor["analysis_model"]["model_id"] == "florence-2-large"
    assert descriptor["execution"]["capture_owned_by_consumer_block"] is True
    assert result["stream_id"] == "stream_123"


def test_florence2_screen_slide_analysis_block_captures_frames_and_emits_slide_artifacts(
    monkeypatch,
    tmp_path,
):
    capture_descriptor = {
        "schema_version": "livepeer.screen_slide_capture.v1",
        "kind": "screen_slide_capture",
        "video_source_descriptor": {
            "kind": "live_video_source",
            "source_type": "vdo_ninja",
            "source": "stream_123",
            "stream_id": "stream_123",
            "password": "",
            "signaling_server": "wss://vdo-signaling-bridge:9443",
            "buffer_ms": 300,
        },
        "source_type": "vdo_ninja",
        "source": "stream_123",
        "stream_id": "stream_123",
        "duration_seconds": 15.0,
        "startup_seconds": 0.0,
        "frame_interval_seconds": 5.0,
        "max_frames": 2,
        "min_slide_gap_seconds": 4.0,
        "slide_change_threshold": 0.72,
    }

    class FakeSegment:
        recording_path = tmp_path / "recording.mp4"
        audio_path = tmp_path / "recording.transcribe.wav"
        log_path = tmp_path / "raspberry.log"

    captured_record_kwargs = {}

    def fake_record_vdo_segment(**kwargs):
        captured_record_kwargs.update(kwargs)
        FakeSegment.recording_path.write_bytes(b"mp4")
        FakeSegment.audio_path.write_bytes(b"wav")
        FakeSegment.log_path.write_text("ok", encoding="utf-8")
        return FakeSegment()

    def fake_extract_video_frames(**kwargs):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_a = frames_dir / "frame_000001.jpg"
        frame_b = frames_dir / "frame_000002.jpg"
        frame_a.write_bytes(b"a")
        frame_b.write_bytes(b"b")
        return [
            {"index": 0, "timestamp_seconds": 0.0, "image_path": str(frame_a)},
            {"index": 1, "timestamp_seconds": 5.0, "image_path": str(frame_b)},
        ]

    monkeypatch.setattr("roboflow_livepeer_blocks.block.record_vdo_segment", fake_record_vdo_segment)
    monkeypatch.setattr("roboflow_livepeer_blocks.block.extract_video_frames", fake_extract_video_frames)
    block = LivepeerFlorence2ScreenSlideAnalysisV1(
        roboflow_api_key="rf-key",
        roboflow_inference_url="http://rf-inference:9001",
        analyzer_cls=FakeFlorenceAnalyzer,
    )

    result = block.run(
        capture_descriptor=capture_descriptor,
        output_dir=str(tmp_path / "visual-analysis"),
    )

    session = result["analysis_session"]
    assert result["status"] == "completed"
    assert result["stream_id"] == "stream_123"
    assert result["frame_count"] == 2
    assert result["slide_count"] == 1
    assert result["slides"][0]["ocr_text"] == "Q3 roadmap\nHiring plan\nBudget update"
    assert result["slides"][0]["chat_text"] == ""
    assert result["visual_events"][0]["slide_changed"] is True
    assert result["visual_events"][1]["slide_changed"] is False
    assert result["visual_events"][1]["slide_text"] == "Q3 roadmap\nHiring plan\nBudget update"
    assert "looks good" in result["visual_events"][1]["chat_text"]
    assert "Mute" in result["visual_events"][1]["call_ui_text"]
    assert "Leave" in result["visual_events"][1]["call_ui_text"]
    assert result["meeting_visual_events"][1]["chat_text"] == result["visual_events"][1]["chat_text"]
    assert result["meeting_visual_summary"]["slide_text"] == "Q3 roadmap\nHiring plan\nBudget update"
    assert "looks good" in result["meeting_visual_summary"]["chat_text"]
    assert result["chat_text"] == result["meeting_visual_summary"]["chat_text"]
    assert captured_record_kwargs["audio_only"] is False
    assert captured_record_kwargs["allow_missing_audio"] is True
    assert session["model_id"] == "florence-2-large"
    assert session["capture_descriptor"] == capture_descriptor
    assert session["meeting_events_jsonl_path"]
    assert Path(result["slides_manifest_path"]).exists()
    assert Path(result["events_jsonl_path"]).exists()
    assert Path(result["meeting_events_jsonl_path"]).exists()
    assert Path(result["result_json_path"]).exists()


def test_livepeer_plugin_loader_registers_visual_blocks():
    from roboflow_livepeer_blocks import load_blocks

    block_names = {block.__name__ for block in load_blocks()}

    assert "LivepeerScreenSlideCaptureV1" in block_names
    assert "LivepeerFlorence2ScreenSlideAnalysisV1" in block_names


def test_florence2_screen_slide_analysis_can_route_to_remote_runner(
    monkeypatch,
    tmp_path,
):
    capture_descriptor = {
        "schema_version": "livepeer.screen_slide_capture.v1",
        "kind": "screen_slide_capture",
        "video_source_descriptor": {
            "kind": "live_video_source",
            "source_type": "vdo_ninja",
            "source": "stream_123",
            "stream_id": "stream_123",
            "password": "",
            "signaling_server": "wss://vdo-signaling-bridge:9443",
            "buffer_ms": 300,
        },
        "source_type": "vdo_ninja",
        "source": "stream_123",
        "stream_id": "stream_123",
        "duration_seconds": 5.0,
        "startup_seconds": 0.0,
        "frame_interval_seconds": 5.0,
        "max_frames": 1,
        "min_slide_gap_seconds": 4.0,
        "slide_change_threshold": 0.72,
    }

    class FakeSegment:
        recording_path = tmp_path / "recording.mp4"
        audio_path = tmp_path / "recording.transcribe.wav"
        log_path = tmp_path / "raspberry.log"

    class CapturingAnalyzer:
        captured = {}

        def __init__(self, **kwargs):
            type(self).captured = kwargs

        def analyze_image(self, image_path, meeting_context_prompt=""):
            return {
                "caption": "Remote slide",
                "detailed_caption": "Remote slide with bullet points",
                "ocr_text": "Quarterly update",
                "meeting_context": {
                    "supported": bool(meeting_context_prompt),
                    "prompt": meeting_context_prompt,
                    "text": meeting_context_prompt,
                    "error": "",
                },
            }

    def fake_record_vdo_segment(**_kwargs):
        FakeSegment.recording_path.write_bytes(b"mp4")
        FakeSegment.audio_path.write_bytes(b"wav")
        FakeSegment.log_path.write_text("ok", encoding="utf-8")
        return FakeSegment()

    def fake_extract_video_frames(**_kwargs):
        frame_path = tmp_path / "frame_000001.jpg"
        frame_path.write_bytes(b"a")
        return [{"index": 0, "timestamp_seconds": 0.0, "image_path": str(frame_path)}]

    monkeypatch.setattr("roboflow_livepeer_blocks.block.record_vdo_segment", fake_record_vdo_segment)
    monkeypatch.setattr("roboflow_livepeer_blocks.block.extract_video_frames", fake_extract_video_frames)

    block = LivepeerFlorence2ScreenSlideAnalysisV1(
        analyzer_cls=CapturingAnalyzer,
        vision_backend="remote",
        florence2_runner_url="https://vision.example",
    )

    result = block.run(
        capture_descriptor=capture_descriptor,
        output_dir=str(tmp_path / "visual-analysis"),
        vision_backend="remote",
        florence2_runner_url="https://vision.example",
    )

    assert CapturingAnalyzer.captured["vision_backend"] == "remote"
    assert CapturingAnalyzer.captured["runner_url"] == "https://vision.example"
    assert result["analysis_session"]["vision_backend"] == "remote"
    assert result["analysis_session"]["remote_transport"] == "direct_runner"
    assert result["analysis_session"]["florence2_runner_url"] == "https://vision.example"


def test_florence2_screen_slide_analysis_can_route_to_livepeer_gateway(
    monkeypatch,
    tmp_path,
):
    capture_descriptor = {
        "schema_version": "livepeer.screen_slide_capture.v1",
        "kind": "screen_slide_capture",
        "video_source_descriptor": {
            "kind": "live_video_source",
            "source_type": "vdo_ninja",
            "source": "stream_123",
            "stream_id": "stream_123",
            "password": "",
            "signaling_server": "wss://vdo-signaling-bridge:9443",
            "buffer_ms": 300,
        },
        "source_type": "vdo_ninja",
        "source": "stream_123",
        "stream_id": "stream_123",
        "duration_seconds": 5.0,
        "startup_seconds": 0.0,
        "frame_interval_seconds": 5.0,
        "max_frames": 1,
        "min_slide_gap_seconds": 4.0,
        "slide_change_threshold": 0.72,
    }

    class FakeSegment:
        recording_path = tmp_path / "recording.mp4"
        audio_path = tmp_path / "recording.transcribe.wav"
        log_path = tmp_path / "raspberry.log"

    class CapturingAnalyzer:
        captured = {}

        def __init__(self, **kwargs):
            type(self).captured = kwargs

        def analyze_image(self, image_path, meeting_context_prompt=""):
            return {
                "caption": "Remote slide",
                "detailed_caption": "Remote slide with bullet points",
                "ocr_text": "Quarterly update",
                "meeting_context": {
                    "supported": bool(meeting_context_prompt),
                    "prompt": meeting_context_prompt,
                    "text": meeting_context_prompt,
                    "error": "",
                },
            }

    def fake_record_vdo_segment(**_kwargs):
        FakeSegment.recording_path.write_bytes(b"mp4")
        FakeSegment.audio_path.write_bytes(b"wav")
        FakeSegment.log_path.write_text("ok", encoding="utf-8")
        return FakeSegment()

    def fake_extract_video_frames(**_kwargs):
        frame_path = tmp_path / "frame_000001.jpg"
        frame_path.write_bytes(b"a")
        return [{"index": 0, "timestamp_seconds": 0.0, "image_path": str(frame_path)}]

    monkeypatch.setattr("roboflow_livepeer_blocks.block.record_vdo_segment", fake_record_vdo_segment)
    monkeypatch.setattr("roboflow_livepeer_blocks.block.extract_video_frames", fake_extract_video_frames)

    block = LivepeerFlorence2ScreenSlideAnalysisV1(
        analyzer_cls=CapturingAnalyzer,
        vision_backend="livepeer_remote",
        florence2_runner_url="https://vision.example",
        livepeer_api_key="lp-key",
        livepeer_base_url="https://loc.example",
        vision_capability="openai:vision",
        vision_offering="florence-2-large",
    )

    result = block.run(
        capture_descriptor=capture_descriptor,
        output_dir=str(tmp_path / "visual-analysis"),
        vision_backend="livepeer_remote",
        florence2_runner_url="https://vision.example",
        livepeer_capability="openai:vision",
        livepeer_offering="florence-2-large",
    )

    assert CapturingAnalyzer.captured["livepeer_api_key"] == "lp-key"
    assert CapturingAnalyzer.captured["livepeer_base_url"] == "https://loc.example"
    assert CapturingAnalyzer.captured["livepeer_capability"] == "openai:vision"
    assert CapturingAnalyzer.captured["livepeer_offering"] == "florence-2-large"
    assert result["analysis_session"]["remote_transport"] == "livepeer_clearinghouse"
    assert result["analysis_session"]["florence2_runner_url"] == ""
    assert result["analysis_session"]["livepeer_capability"] == "openai:vision"
    assert result["analysis_session"]["remote_primary_endpoint"] == "/v1/chat/completions"
    assert result["analysis_session"]["livepeer_offering"] == "florence-2-large"


def test_vdo_direct_true_streaming_session_block_uses_live_pipe_runner(monkeypatch):
    class FakeDirectRunner:
        def run(self):
            return {
                "session_id": "stream_direct",
                "stream_id": "stream_123",
                "status": "closed",
                "source_mode": "vdo_ninja_fdsink_live_pcm",
                "source": "stream_123",
                "sent_audio_seconds": 60.0,
                "sent_frame_count": 750,
                "text": "hello direct stream",
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
                "transcript_events": [{"event_type": "transcript.segment"}],
                "transcript_event_count": 1,
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "/tmp/transcript.txt",
                "publisher_log_path": "/tmp/raspberry.log",
                "ffmpeg_log_path": "/tmp/ffmpeg.log",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeDirectRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_vdo_direct_true_streaming_runner",
        fake_build_runner,
    )
    block = LivepeerVDONinjaDirectTrueStreamingSessionV1(runner_url="http://runner:8080")

    result = block.run(
        source="stream_123",
        signaling_server="wss://vdo-signaling-bridge:9443",
        duration_seconds=60,
        session_id="stream_direct",
    )

    assert captured["runner_url"] == "http://runner:8080"
    assert captured["source"] == "stream_123"
    assert captured["signaling_server"] == "wss://vdo-signaling-bridge:9443"
    assert captured["duration_seconds"] == 60.0
    assert result["source_mode"] == "vdo_ninja_fdsink_live_pcm"
    assert result["sent_audio_seconds"] == 60.0
    assert result["text"] == "hello direct stream"


def test_vdo_direct_true_streaming_remote_fallback_defers_chunk_upload_until_finish(monkeypatch):
    class FakeDirectRunner:
        def run(self):
            return {
                "session_id": "stream_direct",
                "stream_id": "stream_123",
                "status": "closed",
                "source_mode": "vdo_ninja_fdsink_live_pcm",
                "source": "stream_123",
                "sent_audio_seconds": 12.0,
                "sent_frame_count": 150,
                "text": "",
                "speaker_count": 0,
                "speakers": [],
                "transcript_events": [],
                "transcript_event_count": 0,
                "events_jsonl_path": "/tmp/events.jsonl",
                "result_json_path": "/tmp/result.json",
                "transcript_text_path": "",
                "publisher_log_path": "/tmp/raspberry.log",
                "ffmpeg_log_path": "/tmp/ffmpeg.log",
            }

    captured = {}

    def fake_build_runner(**kwargs):
        captured.update(kwargs)
        return FakeDirectRunner()

    monkeypatch.setattr(
        "roboflow_livepeer_blocks.block.build_vdo_direct_true_streaming_runner",
        fake_build_runner,
    )
    block = LivepeerVDONinjaDirectTrueStreamingSessionV1(
        runner_url="http://runner:8080",
        api_key="test-key",
        transcription_backend="livepeer_remote",
    )

    block.run(
        source="stream_123",
        signaling_server="wss://vdo-signaling-bridge:9443",
        duration_seconds=12.0,
        frame_duration_seconds=0.08,
        session_id="stream_direct",
    )

    assert captured["client_init_kwargs"]["chunk_size_seconds"] == 12.08
