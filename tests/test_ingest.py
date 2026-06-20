import subprocess
from pathlib import Path

import pytest

from roboflow_livepeer_blocks import ingest


def test_raspberry_ninja_python_executable_honors_override(monkeypatch):
    monkeypatch.setenv("RASPBERRY_NINJA_PYTHON", "/custom/python")

    assert ingest.raspberry_ninja_python_executable() == "/custom/python"


def test_raspberry_ninja_python_executable_falls_back_to_system_gi(monkeypatch):
    monkeypatch.delenv("RASPBERRY_NINJA_PYTHON", raising=False)
    monkeypatch.setattr(ingest.sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/python3" if name == "python3" else None)
    monkeypatch.setattr(
        ingest,
        "_python_can_import_gi",
        lambda executable: executable == "/usr/bin/python3",
    )

    assert ingest.raspberry_ninja_python_executable() == "/usr/bin/python3"


def test_raspberry_ninja_python_executable_tries_usr_bin_when_path_python_is_venv(monkeypatch):
    monkeypatch.delenv("RASPBERRY_NINJA_PYTHON", raising=False)
    monkeypatch.setattr(ingest.sys, "executable", "/venv/bin/python")
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/venv/bin/python3" if name == "python3" else None)
    monkeypatch.setattr(
        ingest,
        "_python_can_import_gi",
        lambda executable: executable == "/usr/bin/python3",
    )

    assert ingest.raspberry_ninja_python_executable() == "/usr/bin/python3"


def test_parse_vdo_stream_id_accepts_direct_id_and_view_url():
    assert ingest.parse_vdo_stream_id("stream_av53zc79i") == "stream_av53zc79i"
    assert (
        ingest.parse_vdo_stream_id("https://vdo.ninja/?view=stream_av53zc79i")
        == "stream_av53zc79i"
    )


def test_parse_vdo_stream_id_rejects_url_without_stream_id():
    with pytest.raises(ValueError, match="stream id"):
        ingest.parse_vdo_stream_id("https://vdo.ninja/")


def test_resolve_vdo_stream_source_accepts_explicit_stream_id():
    result = ingest.resolve_vdo_stream_source(
        source="stream_123",
        signaling_server="wss://bridge.test:9443",
    )

    assert result["source"] == "stream_123"
    assert result["stream_id"] == "stream_123"
    assert result["signaling_server"] == "wss://bridge.test:9443"
    assert result["auto_resolved"] is False


def test_resolve_vdo_stream_source_marks_explicit_unsuffixed_bridge_stream_unencrypted(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "_read_vdo_bridge_status",
        lambda status_url: {
            "streams": {"stream_123": "uuid"},
            "clients": [{"stream_id": "stream_123", "connected_at_epoch": 1.0}],
        },
    )

    result = ingest.resolve_vdo_stream_source(
        source="wss://bridge.test:9443/?view=stream_123",
        signaling_server="wss://bridge.test:9443",
    )

    assert result["source"] == "wss://bridge.test:9443/?view=stream_123"
    assert result["stream_id"] == "stream_123"
    assert result["signaling_server"] == "wss://bridge.test:9443"
    assert result["password"] == "false"
    assert result["auto_resolved"] is False


def test_resolve_vdo_stream_source_marks_same_bridge_viewer_url_unencrypted_without_status():
    result = ingest.resolve_vdo_stream_source(
        source="wss://bridge.test:9443/?view=stream_123",
        signaling_server="wss://bridge.test:9443",
    )

    assert result["source"] == "wss://bridge.test:9443/?view=stream_123"
    assert result["stream_id"] == "stream_123"
    assert result["signaling_server"] == "wss://bridge.test:9443"
    assert result["password"] == "false"
    assert result["auto_resolved"] is False


def test_resolve_vdo_stream_source_auto_selects_latest_bridge_stream(monkeypatch):
    calls = []

    def fake_read_status(status_url):
        calls.append(status_url)
        return {
            "streams": {
                "stream_old": "uuid-old",
                "stream_new": "uuid-new",
            },
            "clients": [
                {"stream_id": "stream_old", "connected_at_epoch": 100.0},
                {"stream_id": "stream_new", "connected_at_epoch": 200.0},
            ],
        }

    monkeypatch.setattr(ingest, "_read_vdo_bridge_status", fake_read_status)

    result = ingest.resolve_vdo_stream_source(
        source="auto",
        signaling_server="wss://bridge.test:9443",
    )

    assert result["source"] == "stream_new"
    assert result["stream_id"] == "stream_new"
    assert result["signaling_server"] == "wss://bridge.test:9443"
    assert result["status_url"] == "https://bridge.test:9443/statusz"
    assert result["auto_resolved"] is True
    assert calls == ["https://bridge.test:9443/statusz"]


def test_resolve_vdo_stream_source_auto_accepts_joinroom_style_client_status(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "_read_vdo_bridge_status",
        lambda status_url: {
            "clients": [
                {"streamID": "stream_room_old", "connected_at_epoch": 100.0},
                {"streamID": "stream_room_new", "connected_at_epoch": 200.0},
            ],
        },
    )

    result = ingest.resolve_vdo_stream_source(
        source="auto",
        signaling_server="wss://bridge.test:9443",
    )

    assert result["source"] == "stream_room_new"
    assert result["stream_id"] == "stream_room_new"
    assert result["auto_resolved"] is True


def test_resolve_vdo_stream_source_treats_bare_bridge_url_as_auto(monkeypatch):
    called = []

    def fake_read_status(status_url):
        called.append(status_url)
        return {
            "streams": {"stream_from_url": "uuid"},
            "clients": [{"stream_id": "stream_from_url", "connected_at_epoch": 1.0}],
        }

    monkeypatch.setattr(
        ingest,
        "_read_vdo_bridge_status",
        fake_read_status,
    )

    result = ingest.resolve_vdo_stream_source(
        source="wss://bridge.test:9443",
        signaling_server="wss://default-bridge.test:9443",
    )

    assert result["source"] == "stream_from_url"
    assert result["stream_id"] == "stream_from_url"
    assert result["signaling_server"] == "wss://bridge.test:9443"
    assert result["requested_source"] == "auto"
    assert called == ["https://bridge.test:9443/statusz"]


def test_resolve_vdo_stream_source_strips_raspberry_ninja_hash_suffix(monkeypatch):
    assert (
        ingest.raspberry_ninja_hash_suffix(signaling_server="wss://vdo-signaling-bridge:9443")
        == "808d64"
    )
    monkeypatch.setattr(
        ingest,
        "_read_vdo_bridge_status",
        lambda status_url: {
            "streams": {"rf_local_test_2808d64": "uuid"},
            "clients": [{"stream_id": "rf_local_test_2808d64", "connected_at_epoch": 1.0}],
        },
    )

    result = ingest.resolve_vdo_stream_source(
        source="wss://vdo-signaling-bridge:9443",
    )

    assert result["source"] == "rf_local_test_2"
    assert result["stream_id"] == "rf_local_test_2"
    assert result["bridge_stream_id"] == "rf_local_test_2808d64"
    assert result["password"] == ""


def test_resolve_vdo_stream_source_marks_unsuffixed_auto_stream_unencrypted(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "_read_vdo_bridge_status",
        lambda status_url: {
            "streams": {"codexrepro": "uuid"},
            "clients": [{"stream_id": "codexrepro", "connected_at_epoch": 1.0}],
        },
    )

    result = ingest.resolve_vdo_stream_source(
        source="auto",
        signaling_server="wss://vdo-signaling-bridge:9443",
    )

    assert result["source"] == "codexrepro"
    assert result["stream_id"] == "codexrepro"
    assert result["bridge_stream_id"] == "codexrepro"
    assert result["password"] == "false"


def test_capture_rolling_audio_segments_runs_each_segment(monkeypatch, tmp_path):
    calls = []

    def fake_record_vdo_segment(**kwargs):
        calls.append(kwargs)
        index = kwargs["segment_index"]
        return ingest.RollingSegment(
            index=index,
            stream_id="stream_123",
            recording_path=tmp_path / f"seg{index}.webm",
            audio_path=tmp_path / f"seg{index}.wav",
            log_path=tmp_path / f"seg{index}.log",
            started_at_epoch=10.0 + index,
            completed_at_epoch=20.0 + index,
            requested_duration_seconds=kwargs["duration_seconds"],
            audio_duration_seconds=kwargs["duration_seconds"],
        )

    monkeypatch.setattr(ingest, "record_vdo_segment", fake_record_vdo_segment)

    result = ingest.capture_rolling_audio_segments(
        source="stream_123",
        output_dir=tmp_path,
        segment_count=2,
        segment_duration_seconds=5,
        startup_seconds=3,
    )

    assert result["audio_paths"] == [str(tmp_path / "seg0.wav"), str(tmp_path / "seg1.wav")]
    assert result["latest_audio_path"] == str(tmp_path / "seg1.wav")
    assert [call["startup_seconds"] for call in calls] == [3, 0.0]


def test_newest_recording_path_prefers_audio_recording(tmp_path):
    before = set()
    log_path = tmp_path / "raspberry.log"
    log_path.write_text("ok")
    video = tmp_path / "capture.webm"
    audio = tmp_path / "capture_audio.webm"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    assert ingest.newest_recording_path(
        target_dir=tmp_path,
        before=before,
        log_path=log_path,
    ) == audio


def test_newest_recording_path_prefers_video_when_requested(tmp_path):
    before = set()
    log_path = tmp_path / "raspberry.log"
    log_path.write_text("ok")
    video = tmp_path / "capture.ts"
    audio = tmp_path / "capture_audio.webm"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")

    assert ingest.newest_recording_path(
        target_dir=tmp_path,
        before=before,
        log_path=log_path,
        prefer_audio=False,
    ) == video
    assert ingest.newest_recording_path(
        target_dir=tmp_path,
        before=before,
        log_path=log_path,
        prefer_audio=True,
    ) == audio


def test_record_vdo_segment_uses_raspberry_ninja_and_extracts_audio(monkeypatch, tmp_path):
    sleeps = []
    killed = []

    class FakeProcess:
        pid = 12345

        def __init__(self, *args, **kwargs):
            self.command = args[0]
            (tmp_path / "stream_123_seg0000_1_audio.webm").write_bytes(b"audio")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(ingest.time, "time", lambda: 1.0)
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(ingest.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ingest.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(ingest, "raspberry_ninja_python_executable", lambda: "/usr/bin/python3")
    monkeypatch.setattr(ingest, "extract_audio_to_wav", lambda recording_path, audio_path: Path(audio_path).write_bytes(b"wav"))
    monkeypatch.setattr(ingest, "probe_media_duration_seconds", lambda path: 4.5)

    segment = ingest.record_vdo_segment(
        source="stream_123",
        output_dir=tmp_path,
        duration_seconds=4,
        startup_seconds=2,
        buffer_ms=300,
        raspberry_ninja_path=Path("publish.py"),
    )

    assert segment.stream_id == "stream_123"
    assert segment.audio_duration_seconds == 4.5
    assert sleeps == [2, 4]
    assert killed


def test_record_vdo_segment_can_tolerate_missing_audio_for_visual_capture(monkeypatch, tmp_path):
    class FakeProcess:
        pid = 12345

        def __init__(self, *args, **kwargs):
            (tmp_path / "stream_123_seg0000_1.ts").write_bytes(b"video")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(ingest.time, "time", lambda: 1.0)
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(ingest.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ingest.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(ingest, "raspberry_ninja_python_executable", lambda: "/usr/bin/python3")

    def fail_extract(recording_path, audio_path):
        raise subprocess.CalledProcessError(234, ["ffmpeg"])

    monkeypatch.setattr(ingest, "extract_audio_to_wav", fail_extract)

    segment = ingest.record_vdo_segment(
        source="stream_123",
        output_dir=tmp_path,
        duration_seconds=4,
        startup_seconds=2,
        buffer_ms=300,
        audio_only=False,
        allow_missing_audio=True,
        raspberry_ninja_path=Path("publish.py"),
    )

    assert segment.recording_path == tmp_path / "stream_123_seg0000_1.ts"
    assert segment.audio_duration_seconds == 0.0
