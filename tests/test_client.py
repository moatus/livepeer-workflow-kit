from pathlib import Path

from roboflow_livepeer_blocks.audio import AudioChunk
from roboflow_livepeer_blocks.client import (
    ChunkTranscription,
    LivepeerOpenClearinghouseClient,
    aggregate_transcriptions,
)
from roboflow_livepeer_blocks.nemo_client import NemoDiarizedTranscriptionClient


class FakeResponse:
    def __init__(self, body, status_code=200, headers=None):
        self._body = body
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeHttpClient:
    def __init__(self):
        self.calls = []
        self.broker_headers = {"X-Livepeer-Work-Units": "7"}
        self.broker_status = 200
        self.broker_body = {"text": "hello"}

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/v1/jobs"):
            return FakeResponse(
                {
                    "job_id": "job-1",
                    "work_id": "work-1",
                    "broker_url": "https://broker.example",
                    "mode": "handoff",
                    "payment_envelope": "pay",
                    "settle_endpoint": "/v1/jobs/job-1/settle",
                }
            )
        if url.endswith("/v1/cap"):
            return FakeResponse(
                self.broker_body,
                status_code=self.broker_status,
                headers=self.broker_headers,
            )
        if url.endswith("/settle"):
            return FakeResponse({"ok": True})
        raise AssertionError(url)


class FakeNemoHttpClient:
    def __init__(self):
        self.calls = []
        self.status_code = 200
        self.headers = {"X-Livepeer-Work-Units": "9"}
        self.body = {
            "id": "dtx_local_1",
            "text": "speaker_0: hello",
            "segments": [],
            "words": [],
            "usage": {"work_units": 9},
        }
        self.openai_body = {
            "task": "transcribe",
            "language": "english",
            "duration": 1.0,
            "text": "hello",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": "speaker_0",
                }
            ],
            "words": [{"word": "hello", "start": 0.0, "end": 0.5, "speaker": "speaker_0"}],
            "transcription_id": "dtx_local_1",
            "capability": "openai:audio-transcriptions",
            "mode": "local-direct",
            "usage": {"work_units": 9},
            "speaker_labeled_text": "speaker_0: hello",
            "diarization": {
                "speaker_count": 1,
                "speakers": [{"id": "speaker_0", "talk_seconds": 1.0}],
                "segments": [
                    {
                        "speaker": "speaker_0",
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                    }
                ],
                "words": [
                    {"speaker": "speaker_0", "start": 0.0, "end": 0.5, "word": "hello"}
                ],
            },
            "artifacts": {"srt_path": "/tmp/audio.srt"},
        }

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith("/live/sessions"):
            return FakeResponse(
                {"session_id": kwargs["json"].get("session_id", "live_1"), "status": "active"},
                status_code=self.status_code,
                headers=self.headers,
            )
        if "/live/sessions/" in url and url.endswith("/audio"):
            return FakeResponse(
                {"session_id": url.split("/live/sessions/")[1].split("/")[0], "event_type": "audio.ingested"},
                status_code=self.status_code,
                headers=self.headers,
            )
        if "/live/sessions/" in url and url.endswith("/finish"):
            return FakeResponse(
                {"session_id": url.split("/live/sessions/")[1].split("/")[0], "event_type": "session.finished"},
                status_code=self.status_code,
                headers=self.headers,
            )
        if url.endswith("/v1/audio/transcriptions"):
            return FakeResponse(self.openai_body, status_code=self.status_code, headers=self.headers)
        return FakeResponse(self.body, status_code=self.status_code, headers=self.headers)

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse({"session_id": url.rsplit("/", 1)[-1], "status": "active"})


def test_transcribe_chunk_uses_direct_handoff_flow(tmp_path):
    audio_file = tmp_path / "chunk.mp3"
    audio_file.write_bytes(b"audio")
    fake_http = FakeHttpClient()
    client = LivepeerOpenClearinghouseClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        http_client=fake_http,
    )

    result = client.transcribe_chunk(
        AudioChunk(
            index=0,
            path=audio_file,
            start_seconds=0,
            end_seconds=6.1,
            duration_seconds=6.1,
            temporary=False,
        )
    )

    assert result.text == "hello"
    assert result.actual_units == 7
    assert result.job_id == "job-1"
    assert [call[0] for call in fake_http.calls] == [
        "https://loc.cloudspe.com/v1/jobs",
        "https://broker.example/v1/cap",
        "https://loc.cloudspe.com/v1/jobs/job-1/settle",
    ]
    assert fake_http.calls[0][1]["json"]["estimated_units"] == 7
    assert fake_http.calls[0][1]["headers"]["X-API-Key"] == "key"
    assert fake_http.calls[1][1]["headers"]["Livepeer-Capability"] == "openai:audio-transcriptions"
    assert fake_http.calls[1][1]["headers"]["Livepeer-Payment"] == "pay"
    assert fake_http.calls[1][1]["data"] == {
        "model": "whisper-large-v3",
        "response_format": "json",
    }
    assert fake_http.calls[2][1]["json"] == {"actual_units": 7}


def test_nemo_diarized_client_defaults_to_openai_compatible_api_and_preserves_diarization(
    tmp_path,
    monkeypatch,
):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"audio")
    fake_http = FakeNemoHttpClient()
    client = NemoDiarizedTranscriptionClient(
        base_url="http://nemo:8080",
        http_client=fake_http,
    )

    result = client.diarized_transcribe_audio_file(
        str(audio_file),
        num_speakers=2,
        max_speakers=4,
    )

    assert result["text"] == "speaker_0: hello"
    assert result["actual_units"] == 9
    assert result["api_endpoint"] == "/v1/audio/transcriptions"
    assert result["speaker_count"] == 1
    assert result["segments"][0]["speaker"] == "speaker_0"
    assert result["openai_text"] == "hello"
    assert fake_http.calls[0][0] == "http://nemo:8080/v1/audio/transcriptions"
    assert fake_http.calls[0][1]["data"]["language"] == "en"
    assert fake_http.calls[0][1]["data"]["preset"] == "meeting"
    assert fake_http.calls[0][1]["data"]["num_speakers"] == "2"
    assert fake_http.calls[0][1]["data"]["max_speakers"] == "4"
    assert fake_http.calls[0][1]["data"]["response_format"] == "verbose_json"
    assert fake_http.calls[0][1]["data"]["timestamp_granularities[]"] == ["segment", "word"]
    assert fake_http.calls[0][1]["data"]["diarization"] == "true"
    assert "file" in fake_http.calls[0][1]["files"]


def test_nemo_diarized_client_uses_env_default_runner_url(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"audio")
    fake_http = FakeNemoHttpClient()
    monkeypatch.delenv("AUDIO_DIARIZED_TRANSCRIPTION_RUNNER_URL", raising=False)
    monkeypatch.setenv("NEMO_DIARIZED_RUNNER_URL", "http://localhost:18080")
    client = NemoDiarizedTranscriptionClient(http_client=fake_http)

    client.diarized_transcribe_audio_file(str(audio_file))

    assert fake_http.calls[0][0] == "http://localhost:18080/v1/audio/transcriptions"


def test_audio_diarized_client_prefers_standalone_runner_env_url(tmp_path, monkeypatch):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"audio")
    fake_http = FakeNemoHttpClient()
    monkeypatch.setenv("NEMO_DIARIZED_RUNNER_URL", "http://legacy:8080")
    monkeypatch.setenv("AUDIO_DIARIZED_TRANSCRIPTION_RUNNER_URL", "http://standalone:8080")
    client = NemoDiarizedTranscriptionClient(http_client=fake_http)

    client.diarized_transcribe_audio_file(str(audio_file))

    assert fake_http.calls[0][0] == "http://standalone:8080/v1/audio/transcriptions"


def test_nemo_diarized_client_drives_live_session_api(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"audio")
    fake_http = FakeNemoHttpClient()
    client = NemoDiarizedTranscriptionClient(
        base_url="http://nemo:8080",
        http_client=fake_http,
    )

    created = client.create_live_session(
        session_id="live_test",
        vad_strategy="provided",
        rolling_window_seconds=12.0,
    )
    ingested = client.ingest_live_audio_file(
        session_id="live_test",
        audio_path=str(audio_file),
        sequence_index=3,
        vad_segments=[{"start": 0.1, "end": 0.8}],
    )
    snapshot = client.get_live_session("live_test")
    finished = client.finish_live_session(
        session_id="live_test",
        run_final_transcription=True,
        include_words=False,
    )

    assert created["session_id"] == "live_test"
    assert ingested["event_type"] == "audio.ingested"
    assert snapshot["status"] == "active"
    assert finished["event_type"] == "session.finished"
    assert fake_http.calls[0][0] == "http://nemo:8080/v1/audio/diarized-transcriptions/live/sessions"
    assert fake_http.calls[0][1]["json"]["vad_strategy"] == "provided"
    assert fake_http.calls[0][1]["json"]["rolling_window_seconds"] == 12.0
    assert fake_http.calls[1][0] == (
        "http://nemo:8080/v1/audio/diarized-transcriptions/live/sessions/live_test/audio"
    )
    assert fake_http.calls[1][1]["data"]["sequence_index"] == "3"
    assert "vad_segments_json" in fake_http.calls[1][1]["data"]
    assert fake_http.calls[3][1]["json"]["run_final_transcription"] is True
    assert fake_http.calls[3][1]["json"]["include_words"] is False


def test_transcribe_chunk_fails_loudly_when_success_response_has_no_work_units_header(tmp_path):
    audio_file = tmp_path / "chunk.mp3"
    audio_file.write_bytes(b"audio")
    fake_http = FakeHttpClient()
    fake_http.broker_headers = {}
    client = LivepeerOpenClearinghouseClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        http_client=fake_http,
    )

    try:
        client.transcribe_chunk(
            AudioChunk(
                index=0,
                path=audio_file,
                start_seconds=0,
                end_seconds=6.1,
                duration_seconds=6.1,
                temporary=False,
            )
        )
    except RuntimeError as error:
        assert "Livepeer-Work-Units" in str(error)
    else:
        raise AssertionError("Expected RuntimeError when work units header is missing")

    assert fake_http.calls[-1][0] == "https://loc.cloudspe.com/v1/jobs/job-1/settle"
    assert fake_http.calls[-1][1]["json"] == {"actual_units": 0}


def test_transcribe_chunk_settles_zero_units_before_raising_broker_error(tmp_path):
    audio_file = tmp_path / "chunk.mp3"
    audio_file.write_bytes(b"audio")
    fake_http = FakeHttpClient()
    fake_http.broker_status = 507
    fake_http.broker_headers = {}
    fake_http.broker_body = {"detail": {"error": {"message": "GPU out of memory"}}}
    client = LivepeerOpenClearinghouseClient(
        api_key="key",
        base_url="https://loc.cloudspe.com",
        http_client=fake_http,
    )

    try:
        client.transcribe_chunk(
            AudioChunk(
                index=0,
                path=audio_file,
                start_seconds=0,
                end_seconds=6.1,
                duration_seconds=6.1,
                temporary=False,
            )
        )
    except RuntimeError as error:
        assert "status 507" in str(error)
    else:
        raise AssertionError("Expected RuntimeError for broker failure")

    assert fake_http.calls[-1][0] == "https://loc.cloudspe.com/v1/jobs/job-1/settle"
    assert fake_http.calls[-1][1]["json"] == {"actual_units": 0}


def test_aggregate_transcriptions_stitches_ordered_text():
    first = ChunkTranscription(
        chunk=AudioChunk(1, Path("b.mp3"), 10, 20, 10, True),
        text="world",
        actual_units=3,
        job_id="job-b",
        work_id="work-b",
        raw_responses={"b": True},
    )
    second = ChunkTranscription(
        chunk=AudioChunk(0, Path("a.mp3"), 0, 10, 10, True),
        text="hello",
        actual_units=2,
        job_id="job-a",
        work_id="work-a",
        raw_responses={"a": True},
    )

    aggregate = aggregate_transcriptions([first, second], source_audio_path="/tmp/source.mp3")

    assert aggregate["text"] == "hello world"
    assert aggregate["actual_units"] == 5
    assert aggregate["job_ids"] == ["job-a", "job-b"]
    assert aggregate["work_ids"] == ["work-a", "work-b"]
    assert [chunk["index"] for chunk in aggregate["chunks"]] == [0, 1]
    assert [chunk["audio_path"] for chunk in aggregate["chunks"]] == [
        "/tmp/source.mp3",
        "/tmp/source.mp3",
    ]
    assert [chunk["chunk_file_path"] for chunk in aggregate["chunks"]] == [None, None]
