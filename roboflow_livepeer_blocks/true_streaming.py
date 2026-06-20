"""True-streaming WebSocket client and bounded Roboflow session adapter."""

from __future__ import annotations

import inspect
import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol
from urllib.parse import urlencode, urlparse, urlunparse

from .config import (
    init_local_audio_ingest_url,
    init_nemo_diarized_runner_url,
    init_vdo_signaling_server_url,
)
from .ingest import (
    DEFAULT_INGEST_OUTPUT_DIR,
    DEFAULT_RASPBERRY_NINJA,
    _raspberry_env,
    parse_vdo_stream_id,
    raspberry_ninja_python_executable,
    resolve_vdo_stream_source,
    safe_file_prefix,
)
from .local_ingest import parse_local_audio_ingest_source
from .streaming import AudioSegmentSource, LivepeerVDONinjaAudioSegmentSource


@dataclass(frozen=True)
class LivepeerTrueStreamingSessionConfig:
    session_id: str = field(default_factory=lambda: f"stream_{uuid.uuid4().hex[:12]}")
    language: str = "en"
    preset: str = "meeting"
    max_speakers: int = 4
    sample_rate: int = 16000
    frame_duration_seconds: float = 0.08
    artifact_root: str = str(DEFAULT_INGEST_OUTPUT_DIR / "true-streaming")

    def __post_init__(self) -> None:
        if self.max_speakers <= 0:
            raise ValueError("max_speakers must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_duration_seconds <= 0:
            raise ValueError("frame_duration_seconds must be positive")


def _build_transport_client(
    *,
    client_cls: Any,
    base_url: Optional[str],
    client_init_kwargs: Optional[Dict[str, Any]] = None,
    initial_receive_timeout_seconds: Optional[float] = None,
) -> Any:
    kwargs: Dict[str, Any] = {"base_url": base_url, **(client_init_kwargs or {})}
    if (
        initial_receive_timeout_seconds is not None
        and "initial_receive_timeout_seconds" not in kwargs
    ):
        try:
            parameters = inspect.signature(client_cls.__init__).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_timeout = any(
            parameter.name == "initial_receive_timeout_seconds"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_timeout:
            kwargs["initial_receive_timeout_seconds"] = initial_receive_timeout_seconds
    return client_cls(**kwargs)


class TrueStreamingTransportSession(Protocol):
    events: List[Dict[str, Any]]

    def __enter__(self) -> "TrueStreamingTransportSession":
        ...

    def __exit__(self, *_: Any) -> None:
        ...

    def send_audio_file(
        self,
        audio_path: str | Path,
        *,
        source_segment: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def send_audio_frame(
        self,
        frame: bytes,
        *,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

    def finish(self) -> List[Dict[str, Any]]:
        ...


class NemoTrueStreamingWebSocketClient:
    """Client for the runner's persistent PCM16 true-streaming WebSocket."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        websocket_connect: Any = None,
        receive_timeout_seconds: float = 0.05,
        initial_receive_timeout_seconds: Optional[float] = None,
        finish_receive_timeout_seconds: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or init_nemo_diarized_runner_url()).rstrip("/")
        self._websocket_connect = websocket_connect
        self.receive_timeout_seconds = receive_timeout_seconds
        self.initial_receive_timeout_seconds = (
            initial_receive_timeout_seconds
            if initial_receive_timeout_seconds is not None
            else 30.0
        )
        self.finish_receive_timeout_seconds = (
            finish_receive_timeout_seconds
            if finish_receive_timeout_seconds is not None
            else self.initial_receive_timeout_seconds
        )

    def connect_session(
        self,
        *,
        session_id: str,
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        sample_rate: int = 16000,
        frame_duration_seconds: float = 0.08,
    ) -> TrueStreamingTransportSession:
        return _NemoTrueStreamingWebSocketSession(
            url=_true_streaming_ws_url(
                self.base_url,
                session_id=session_id,
                language=language,
                preset=preset,
                max_speakers=max_speakers,
                sample_rate=sample_rate,
            ),
            websocket_connect=self._websocket_connect,
            receive_timeout_seconds=self.receive_timeout_seconds,
            initial_receive_timeout_seconds=self.initial_receive_timeout_seconds,
            finish_receive_timeout_seconds=self.finish_receive_timeout_seconds,
            sample_rate=sample_rate,
            frame_duration_seconds=frame_duration_seconds,
        )


class _NemoTrueStreamingWebSocketSession:
    def __init__(
        self,
        *,
        url: str,
        websocket_connect: Any,
        receive_timeout_seconds: float,
        initial_receive_timeout_seconds: float,
        finish_receive_timeout_seconds: float,
        sample_rate: int,
        frame_duration_seconds: float,
    ) -> None:
        self.url = url
        self.receive_timeout_seconds = receive_timeout_seconds
        self.initial_receive_timeout_seconds = initial_receive_timeout_seconds
        self.finish_receive_timeout_seconds = finish_receive_timeout_seconds
        self.sample_rate = sample_rate
        self.frame_duration_seconds = frame_duration_seconds
        self.events: List[Dict[str, Any]] = []
        self._connect = websocket_connect or _load_websocket_connect()
        self._websocket: Any = None
        self._websocket_context: Any = None
        self._closed = False

    def __enter__(self) -> "_NemoTrueStreamingWebSocketSession":
        self._websocket_context = _connect_without_client_pings(self._connect, self.url)
        self._websocket = self._websocket_context
        if hasattr(self._websocket_context, "__enter__"):
            self._websocket = self._websocket_context.__enter__()
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            if not self._closed:
                self.finish()
        finally:
            if self._websocket_context is not None and hasattr(self._websocket_context, "__exit__"):
                self._websocket_context.__exit__(None, None, None)

    def send_audio_file(
        self,
        audio_path: str | Path,
        *,
        source_segment: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        start_index = len(self.events)
        for frame in iter_pcm16_wav_frames(
            audio_path,
            sample_rate=self.sample_rate,
            frame_duration_seconds=self.frame_duration_seconds,
        ):
            self._websocket.send(frame)
            self._drain_received_events()
        emitted = self.events[start_index:]
        if source_segment:
            for event in emitted:
                event.setdefault("source_segment", source_segment)
        return emitted

    def send_audio_frame(
        self,
        frame: bytes,
        *,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        start_index = len(self.events)
        self._websocket.send(frame)
        self._drain_received_events()
        emitted = self.events[start_index:]
        if source_event:
            for event in emitted:
                event.setdefault("source_event", source_event)
        return emitted

    def finish(self) -> List[Dict[str, Any]]:
        if self._closed:
            return []
        start_index = len(self.events)
        try:
            self._websocket.send(json.dumps({"type": "finish"}))
            self._drain_received_events(
                until_finished=True,
                receive_timeout_seconds=self.finish_receive_timeout_seconds,
            )
            if not self._has_finished_event():
                self._append_synthetic_finished_event(
                    finish_reason="finish_ack_timeout",
                )
        except Exception as error:
            if not _is_websocket_closed_error(error):
                raise
            self._append_synthetic_finished_event(
                finish_reason="websocket_closed_before_finish_ack",
                error=error,
            )
        finally:
            self._closed = True
        return self.events[start_index:]

    def _has_finished_event(self) -> bool:
        return any(event.get("event_type") == "transcript.session.finished" for event in self.events)

    def _append_synthetic_finished_event(
        self,
        *,
        finish_reason: str,
        error: Optional[BaseException] = None,
    ) -> None:
        if self._has_finished_event():
            return
        event = {
            "event_type": "transcript.session.finished",
            "session_id": "",
            "status": "closed",
            "finish_reason": finish_reason,
        }
        if error is not None:
            event["transport_error_type"] = error.__class__.__name__
            event["transport_error"] = str(error)
        self.events.append(event)

    def _drain_received_events(
        self,
        *,
        minimum: int = 0,
        until_closed: bool = False,
        until_finished: bool = False,
        receive_timeout_seconds: Optional[float] = None,
    ) -> None:
        received = 0
        saw_finished = False
        timeout = (
            receive_timeout_seconds
            if receive_timeout_seconds is not None
            else self.receive_timeout_seconds
        )
        while True:
            try:
                raw = self._websocket.recv(timeout=timeout)
            except TypeError:
                raw = self._websocket.recv()
            except (TimeoutError, EOFError, StopIteration):
                if until_finished and saw_finished:
                    return
                if until_closed or received >= minimum:
                    return
                raise
            except Exception as error:
                if until_finished and saw_finished:
                    return
                if until_closed or received >= minimum:
                    return
                raise error
            if raw is None:
                if until_finished and saw_finished:
                    return
                if until_closed or received >= minimum:
                    return
                continue
            event = _json_event(raw)
            if event:
                self.events.append(event)
                received += 1
                if event.get("event_type") == "transcript.session.finished":
                    saw_finished = True
            if until_finished and saw_finished:
                continue
            if not until_closed and not until_finished and minimum > 0 and received >= minimum:
                return


class LocalAudioIngestWebSocketClient:
    """Client for consuming localhost audio ingest sessions."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        websocket_connect: Any = None,
    ) -> None:
        self.base_url = (base_url or init_local_audio_ingest_url()).rstrip("/")
        self._websocket_connect = websocket_connect

    def connect_session(self, *, source: str) -> "_LocalAudioIngestConsumerSession":
        info = parse_local_audio_ingest_source(source=source, default_base_url=self.base_url)
        return _LocalAudioIngestConsumerSession(
            source=source,
            session_id=info["session_id"],
            consume_url=info["consume_url"],
            status_url=info["status_url"],
            websocket_connect=self._websocket_connect,
        )


class _LocalAudioIngestConsumerSession:
    def __init__(
        self,
        *,
        source: str,
        session_id: str,
        consume_url: str,
        status_url: str,
        websocket_connect: Any,
    ) -> None:
        self.source = source
        self.session_id = session_id
        self.consume_url = consume_url
        self.status_url = status_url
        self.events: List[Dict[str, Any]] = []
        self._connect = websocket_connect or _load_websocket_connect()
        self._websocket: Any = None
        self._websocket_context: Any = None
        self._closed = False

    def __enter__(self) -> "_LocalAudioIngestConsumerSession":
        self._websocket_context = self._connect(self.consume_url)
        self._websocket = self._websocket_context
        if hasattr(self._websocket_context, "__enter__"):
            self._websocket = self._websocket_context.__enter__()
        return self

    def __exit__(self, *_: Any) -> None:
        self._closed = True
        if self._websocket_context is not None and hasattr(self._websocket_context, "__exit__"):
            self._websocket_context.__exit__(None, None, None)

    def receive(self, *, timeout: Optional[float] = None) -> Optional[object]:
        try:
            raw = self._websocket.recv(timeout=timeout)
        except TypeError:
            raw = self._websocket.recv()
        except (TimeoutError, EOFError, StopIteration):
            return None
        except Exception:
            if self._closed:
                return None
            raise
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw)
        event = _json_event(raw)
        if event:
            self.events.append(event)
            if event.get("event_type") == "source.closed":
                self._closed = True
        return event or None

    @property
    def closed(self) -> bool:
        return self._closed


class LivepeerTrueStreamingSessionRunner:
    """Run one bounded source capture through a persistent true-streaming session."""

    def __init__(
        self,
        *,
        stream_source: AudioSegmentSource,
        client: Any,
        session_config: LivepeerTrueStreamingSessionConfig,
        artifact_dir: str | Path,
    ) -> None:
        self.stream_source = stream_source
        self.client = client
        self.session_config = session_config
        self.artifact_dir = Path(artifact_dir)
        self.events_path = self.artifact_dir / "true-streaming-events.jsonl"
        self.result_path = self.artifact_dir / "true-streaming-session-result.json"
        self.transcript_text_path = self.artifact_dir / "true-streaming-transcript.txt"

    def run(self) -> Dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        events: List[Dict[str, Any]] = []
        captured_segments: List[Dict[str, Any]] = []
        self.stream_source.open()
        try:
            with self.client.connect_session(
                session_id=self.session_config.session_id,
                language=self.session_config.language,
                preset=self.session_config.preset,
                max_speakers=self.session_config.max_speakers,
                sample_rate=self.session_config.sample_rate,
                frame_duration_seconds=self.session_config.frame_duration_seconds,
            ) as session:
                self._append_events(events, getattr(session, "events", []))
                for segment in self.stream_source.segments():
                    segment_payload = segment.as_dict()
                    captured_segments.append(segment_payload)
                    self._append_event(
                        events,
                        {
                            "event_type": "source.audio_chunk",
                            "session_id": self.session_config.session_id,
                            "stream_id": segment.stream_id,
                            "source_segment": segment_payload,
                        },
                    )
                    emitted = session.send_audio_file(
                        segment.audio_path,
                        source_segment=segment_payload,
                    )
                    self._append_events(events, emitted)
                self._append_events(events, session.finish())
        finally:
            self.stream_source.close()

        result = _true_streaming_result_payload(
            session_id=self.session_config.session_id,
            stream_id=str(captured_segments[0].get("stream_id") if captured_segments else ""),
            captured_segments=captured_segments,
            events=events,
            events_path=self.events_path,
            result_path=self.result_path,
            transcript_text_path=self.transcript_text_path,
        )
        if result["text"]:
            self.transcript_text_path.write_text(result["text"] + "\n", encoding="utf-8")
        result["transcript_text_path"] = str(self.transcript_text_path) if result["text"] else ""
        self.result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    def _append_events(self, events: List[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]) -> None:
        for event in incoming:
            self._append_event(events, event)

    def _append_event(self, events: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
        event_with_time = {"roboflow_recorded_at_epoch": time.time(), **event}
        events.append(event_with_time)
        with self.events_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event_with_time, sort_keys=True) + "\n")


class LivepeerVDONinjaDirectTrueStreamingRunner:
    """Pipe live VDO.Ninja audio directly into the persistent runner WebSocket."""

    def __init__(
        self,
        *,
        source: str,
        client: Any,
        session_config: LivepeerTrueStreamingSessionConfig,
        artifact_dir: str | Path,
        duration_seconds: float,
        startup_timeout_seconds: float,
        password: str,
        signaling_server: str,
        buffer_ms: int,
        raspberry_ninja_path: str | Path = DEFAULT_RASPBERRY_NINJA,
    ) -> None:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if buffer_ms < 10:
            raise ValueError("buffer_ms must be at least 10")
        self.source = source
        self.stream_id = parse_vdo_stream_id(source)
        self.client = client
        self.session_config = session_config
        self.artifact_dir = Path(artifact_dir)
        self.duration_seconds = float(duration_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.password = password
        self.signaling_server = signaling_server
        self.buffer_ms = int(buffer_ms)
        self.raspberry_ninja_path = Path(raspberry_ninja_path)
        self.events_path = self.artifact_dir / "direct-true-streaming-events.jsonl"
        self.result_path = self.artifact_dir / "direct-true-streaming-session-result.json"
        self.transcript_text_path = self.artifact_dir / "direct-true-streaming-transcript.txt"
        self.publisher_log_path = self.artifact_dir / "raspberry-ninja-fdsink.log"
        self.ffmpeg_log_path = self.artifact_dir / "ffmpeg-pcm-resample.log"

    def run(self) -> Dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        events: List[Dict[str, Any]] = []
        frame_bytes = max(
            2,
            int(self.session_config.sample_rate * self.session_config.frame_duration_seconds) * 2,
        )
        if frame_bytes % 2:
            frame_bytes += 1
        sent_frame_count = 0
        sent_audio_bytes = 0
        started_at_epoch = time.time()
        source_started_at_epoch: Optional[float] = None
        source_completed_at_epoch: Optional[float] = None

        with self.client.connect_session(
            session_id=self.session_config.session_id,
            language=self.session_config.language,
            preset=self.session_config.preset,
            max_speakers=self.session_config.max_speakers,
            sample_rate=self.session_config.sample_rate,
            frame_duration_seconds=self.session_config.frame_duration_seconds,
        ) as session:
            self._append_events(events, getattr(session, "events", []))
            publisher = None
            ffmpeg = None
            publisher_log = None
            ffmpeg_log = None
            raw_read_fd: Optional[int] = None
            try:
                publisher_log = self.publisher_log_path.open("w", encoding="utf-8")
                ffmpeg_log = self.ffmpeg_log_path.open("w", encoding="utf-8")
                raw_read_fd, publisher = self._start_publisher(publisher_log)
                ffmpeg = self._start_resampler(raw_read_fd, ffmpeg_log)
                raw_read_fd = None
                source_started_at_epoch = time.time()
                self._append_event(
                    events,
                    {
                        "event_type": "source.connected",
                        "source_mode": "vdo_ninja_fdsink_live_pcm",
                        "session_id": self.session_config.session_id,
                        "stream_id": self.stream_id,
                        "source": self.source,
                        "target_audio_sample_rate": self.session_config.sample_rate,
                        "target_frame_duration_seconds": self.session_config.frame_duration_seconds,
                        "started_at_epoch": source_started_at_epoch,
                    },
                )
                sent_frame_count, sent_audio_bytes = self._pump_resampled_pcm(
                    session=session,
                    events=events,
                    ffmpeg=ffmpeg,
                    publisher=publisher,
                    frame_bytes=frame_bytes,
                )
                source_completed_at_epoch = time.time()
                self._append_event(
                    events,
                    {
                        "event_type": "source.audio_finished",
                        "source_mode": "vdo_ninja_fdsink_live_pcm",
                        "session_id": self.session_config.session_id,
                        "stream_id": self.stream_id,
                        "completed_at_epoch": source_completed_at_epoch,
                        "sent_frame_count": sent_frame_count,
                        "sent_audio_bytes": sent_audio_bytes,
                        "sent_audio_seconds": round(
                            sent_audio_bytes / 2 / self.session_config.sample_rate,
                            6,
                        ),
                    },
                )
            finally:
                self._terminate_process(ffmpeg)
                self._terminate_process(publisher)
                if raw_read_fd is not None:
                    try:
                        os.close(raw_read_fd)
                    except OSError:
                        pass
                if ffmpeg_log is not None:
                    ffmpeg_log.close()
                if publisher_log is not None:
                    publisher_log.close()
            self._append_events(events, session.finish())

        result = _true_streaming_result_payload(
            session_id=self.session_config.session_id,
            stream_id=self.stream_id,
            captured_segments=[],
            events=events,
            events_path=self.events_path,
            result_path=self.result_path,
            transcript_text_path=self.transcript_text_path,
        )
        result.update(
            {
                "source_mode": "vdo_ninja_fdsink_live_pcm",
                "source": self.source,
                "started_at_epoch": started_at_epoch,
                "source_started_at_epoch": source_started_at_epoch,
                "source_completed_at_epoch": source_completed_at_epoch,
                "requested_duration_seconds": self.duration_seconds,
                "sent_frame_count": sent_frame_count,
                "sent_audio_bytes": sent_audio_bytes,
                "sent_audio_seconds": round(
                    sent_audio_bytes / 2 / self.session_config.sample_rate,
                    6,
                ),
                "publisher_log_path": str(self.publisher_log_path),
                "ffmpeg_log_path": str(self.ffmpeg_log_path),
            }
        )
        self._write_jsonl(self.events_path, events)
        self.result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        if result["text"]:
            self.transcript_text_path.write_text(result["text"] + "\n", encoding="utf-8")
        result["transcript_text_path"] = str(self.transcript_text_path) if result["text"] else ""
        return result

    def _start_publisher(self, log_file: Any) -> tuple[int, subprocess.Popen[Any]]:
        raw_read_fd, raw_write_fd = os.pipe()
        env = _raspberry_env()
        env["RN_FDSINK_FD"] = str(raw_write_fd)
        command = [
            raspberry_ninja_python_executable(),
            str(self.raspberry_ninja_path),
            "--fdsink",
            self.stream_id,
            "--buffer",
            str(self.buffer_ms),
            "--novideo",
        ]
        if self.password:
            command.extend(["--password", self.password])
        if self.signaling_server:
            command.extend(["--server", self.signaling_server])
        process = subprocess.Popen(
            command,
            cwd=str(self.artifact_dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            pass_fds=(raw_write_fd,),
            close_fds=True,
            start_new_session=True,
        )
        os.close(raw_write_fd)
        return raw_read_fd, process

    def _start_resampler(self, raw_read_fd: int, log_file: Any) -> subprocess.Popen[Any]:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            str(self.session_config.sample_rate),
            "-ac",
            "1",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdin=raw_read_fd,
            stdout=subprocess.PIPE,
            stderr=log_file,
            close_fds=True,
            start_new_session=True,
        )
        os.close(raw_read_fd)
        return process

    def _pump_resampled_pcm(
        self,
        *,
        session: TrueStreamingTransportSession,
        events: List[Dict[str, Any]],
        ffmpeg: subprocess.Popen[Any],
        publisher: subprocess.Popen[Any],
        frame_bytes: int,
    ) -> tuple[int, int]:
        if ffmpeg.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe was not created")
        fd = ffmpeg.stdout.fileno()
        deadline = time.monotonic() + self.startup_timeout_seconds
        target_bytes: Optional[int] = None
        if self.duration_seconds > 0:
            target_bytes = int(self.duration_seconds * self.session_config.sample_rate) * 2
        buffer = b""
        sent_frame_count = 0
        sent_audio_bytes = 0
        first_audio_seen = False
        last_audio_seen_at = 0.0
        while target_bytes is None or sent_audio_bytes < target_bytes:
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                now = time.monotonic()
                if ffmpeg.poll() is not None:
                    if target_bytes is None and sent_audio_bytes > 0:
                        break
                    raise RuntimeError(
                        "ffmpeg exited before producing requested live VDO audio "
                        f"(exit={ffmpeg.returncode}). {self._audio_failure_context()}"
                    )
                if publisher.poll() is not None:
                    if target_bytes is None and sent_audio_bytes > 0:
                        break
                    raise RuntimeError(
                        "Raspberry.Ninja exited before producing requested live VDO audio "
                        f"(exit={publisher.returncode}). {self._audio_failure_context()}"
                    )
                if not first_audio_seen and now > deadline:
                    raise RuntimeError(
                        f"timed out after {self.startup_timeout_seconds:.1f}s waiting for live VDO audio. "
                        f"{self._audio_failure_context()}"
                    )
                if (
                    first_audio_seen
                    and now - last_audio_seen_at > self.startup_timeout_seconds
                    and sent_audio_bytes > 0
                ):
                    self._append_event(
                        events,
                        {
                            "event_type": "source.audio_idle_timeout",
                            "source_mode": "vdo_ninja_fdsink_live_pcm",
                            "session_id": self.session_config.session_id,
                            "stream_id": self.stream_id,
                            "timeout_seconds": self.startup_timeout_seconds,
                            "sent_frame_count": sent_frame_count,
                            "sent_audio_bytes": sent_audio_bytes,
                            "sent_audio_seconds": round(
                                sent_audio_bytes / 2 / self.session_config.sample_rate,
                                6,
                            ),
                        },
                    )
                    break
                continue
            chunk = os.read(fd, max(frame_bytes - len(buffer), 1))
            if not chunk:
                if sent_audio_bytes <= 0:
                    raise RuntimeError(
                        "live VDO audio pipe closed before any PCM was received. "
                        f"{self._audio_failure_context()}"
                    )
                break
            first_audio_seen = True
            last_audio_seen_at = time.monotonic()
            buffer += chunk
            while len(buffer) >= frame_bytes and (
                target_bytes is None or sent_audio_bytes < target_bytes
            ):
                frame = buffer[:frame_bytes]
                buffer = buffer[frame_bytes:]
                if target_bytes is not None:
                    remaining = target_bytes - sent_audio_bytes
                    if len(frame) > remaining:
                        frame = frame[:remaining]
                emitted = session.send_audio_frame(
                    frame,
                    source_event={
                        "source_mode": "vdo_ninja_fdsink_live_pcm",
                        "stream_id": self.stream_id,
                        "frame_index": sent_frame_count,
                    },
                )
                self._append_events(events, emitted)
                sent_frame_count += 1
                sent_audio_bytes += len(frame)
        return sent_frame_count, sent_audio_bytes

    def _audio_failure_context(self) -> str:
        log_tail = ""
        if self.publisher_log_path.exists():
            log_tail = self.publisher_log_path.read_text(errors="replace")[-1200:].strip()
        return (
            f"source={self.source!r}, stream_id={self.stream_id!r}, "
            f"signaling_server={self.signaling_server!r}, "
            f"publisher_log_path={str(self.publisher_log_path)!r}, "
            f"ffmpeg_log_path={str(self.ffmpeg_log_path)!r}, "
            f"publisher_log_tail={log_tail!r}"
        )

    @staticmethod
    def _terminate_process(process: Optional[subprocess.Popen[Any]]) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5)
            return
        except Exception:
            pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            return
        except Exception:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except Exception:
            pass

    @staticmethod
    def _append_event(events: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
        events.append({"roboflow_recorded_at_epoch": time.time(), **event})

    @staticmethod
    def _append_events(events: List[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]) -> None:
        for event in incoming:
            LivepeerVDONinjaDirectTrueStreamingRunner._append_event(events, event)

    @staticmethod
    def _write_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, sort_keys=True) + "\n")


class LivepeerLocalAudioIngressTrueStreamingRunner:
    """Consume a localhost ingest session and forward it into the runner WebSocket."""

    def __init__(
        self,
        *,
        source: str,
        client: Any,
        session_config: LivepeerTrueStreamingSessionConfig,
        artifact_dir: str | Path,
        duration_seconds: float,
        startup_timeout_seconds: float,
        ingest_client: Any,
    ) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        self.source = source
        self.ingest_info = parse_local_audio_ingest_source(source=source)
        self.client = client
        self.session_config = session_config
        self.artifact_dir = Path(artifact_dir)
        self.duration_seconds = float(duration_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.ingest_client = ingest_client
        self.events_path = self.artifact_dir / "local-ingest-true-streaming-events.jsonl"
        self.result_path = self.artifact_dir / "local-ingest-true-streaming-session-result.json"
        self.transcript_text_path = self.artifact_dir / "local-ingest-true-streaming-transcript.txt"

    def run(self) -> Dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        events: List[Dict[str, Any]] = []
        frame_bytes = max(
            2,
            int(self.session_config.sample_rate * self.session_config.frame_duration_seconds) * 2,
        )
        if frame_bytes % 2:
            frame_bytes += 1
        sent_frame_count = 0
        sent_audio_bytes = 0
        started_at_epoch = time.time()
        source_started_at_epoch: Optional[float] = None
        source_completed_at_epoch: Optional[float] = None

        with self.client.connect_session(
            session_id=self.session_config.session_id,
            language=self.session_config.language,
            preset=self.session_config.preset,
            max_speakers=self.session_config.max_speakers,
            sample_rate=self.session_config.sample_rate,
            frame_duration_seconds=self.session_config.frame_duration_seconds,
        ) as session:
            self._append_events(events, getattr(session, "events", []))
            with self.ingest_client.connect_session(source=self.source) as ingest_session:
                source_started_at_epoch = time.time()
                self._append_event(
                    events,
                    {
                        "event_type": "source.connected",
                        "source_mode": "localhost_ingest_pcm",
                        "session_id": self.session_config.session_id,
                        "stream_id": self.ingest_info["session_id"],
                        "source": self.source,
                        "status_url": ingest_session.status_url,
                        "consume_url": ingest_session.consume_url,
                        "target_audio_sample_rate": self.session_config.sample_rate,
                        "target_frame_duration_seconds": self.session_config.frame_duration_seconds,
                        "started_at_epoch": source_started_at_epoch,
                    },
                )
                sent_frame_count, sent_audio_bytes = self._pump_ingest_pcm(
                    session=session,
                    ingest_session=ingest_session,
                    events=events,
                    frame_bytes=frame_bytes,
                )
                source_completed_at_epoch = time.time()
                self._append_event(
                    events,
                    {
                        "event_type": "source.audio_finished",
                        "source_mode": "localhost_ingest_pcm",
                        "session_id": self.session_config.session_id,
                        "stream_id": self.ingest_info["session_id"],
                        "completed_at_epoch": source_completed_at_epoch,
                        "sent_frame_count": sent_frame_count,
                        "sent_audio_bytes": sent_audio_bytes,
                        "sent_audio_seconds": round(
                            sent_audio_bytes / 2 / self.session_config.sample_rate,
                            6,
                        ),
                    },
                )
            self._append_events(events, session.finish())

        result = _true_streaming_result_payload(
            session_id=self.session_config.session_id,
            stream_id=self.ingest_info["session_id"],
            captured_segments=[],
            events=events,
            events_path=self.events_path,
            result_path=self.result_path,
            transcript_text_path=self.transcript_text_path,
        )
        result.update(
            {
                "source_mode": "localhost_ingest_pcm",
                "source": self.source,
                "started_at_epoch": started_at_epoch,
                "source_started_at_epoch": source_started_at_epoch,
                "source_completed_at_epoch": source_completed_at_epoch,
                "requested_duration_seconds": self.duration_seconds,
                "sent_frame_count": sent_frame_count,
                "sent_audio_bytes": sent_audio_bytes,
                "sent_audio_seconds": round(
                    sent_audio_bytes / 2 / self.session_config.sample_rate,
                    6,
                ),
                "ingest_status_url": self.ingest_info["status_url"],
                "ingest_consume_url": self.ingest_info["consume_url"],
                "publisher_log_path": "",
                "ffmpeg_log_path": "",
            }
        )
        self._write_jsonl(self.events_path, events)
        self.result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        if result["text"]:
            self.transcript_text_path.write_text(result["text"] + "\n", encoding="utf-8")
        result["transcript_text_path"] = str(self.transcript_text_path) if result["text"] else ""
        return result

    def _pump_ingest_pcm(
        self,
        *,
        session: TrueStreamingTransportSession,
        ingest_session: _LocalAudioIngestConsumerSession,
        events: List[Dict[str, Any]],
        frame_bytes: int,
    ) -> tuple[int, int]:
        deadline = time.monotonic() + self.startup_timeout_seconds
        target_bytes = int(self.duration_seconds * self.session_config.sample_rate) * 2
        buffer = b""
        sent_frame_count = 0
        sent_audio_bytes = 0
        first_audio_seen = False
        while sent_audio_bytes < target_bytes:
            payload = ingest_session.receive(timeout=1.0)
            if payload is None:
                if ingest_session.closed:
                    break
                if not first_audio_seen and time.monotonic() > deadline:
                    raise RuntimeError(
                        f"timed out after {self.startup_timeout_seconds:.1f}s waiting for local ingest audio"
                    )
                continue
            if isinstance(payload, dict):
                self._append_event(events, payload)
                if payload.get("event_type") == "source.closed":
                    break
                continue
            first_audio_seen = True
            buffer += payload
            while len(buffer) >= frame_bytes and sent_audio_bytes < target_bytes:
                frame = buffer[:frame_bytes]
                buffer = buffer[frame_bytes:]
                remaining = target_bytes - sent_audio_bytes
                if len(frame) > remaining:
                    frame = frame[:remaining]
                emitted = session.send_audio_frame(
                    frame,
                    source_event={
                        "source_mode": "localhost_ingest_pcm",
                        "stream_id": self.ingest_info["session_id"],
                        "frame_index": sent_frame_count,
                    },
                )
                self._append_events(events, emitted)
                sent_frame_count += 1
                sent_audio_bytes += len(frame)
        return sent_frame_count, sent_audio_bytes

    @staticmethod
    def _append_event(events: List[Dict[str, Any]], event: Dict[str, Any]) -> None:
        events.append({"roboflow_recorded_at_epoch": time.time(), **event})

    @staticmethod
    def _append_events(events: List[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]) -> None:
        for event in incoming:
            LivepeerLocalAudioIngressTrueStreamingRunner._append_event(events, event)

    @staticmethod
    def _write_jsonl(path: Path, events: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as output:
            for event in events:
                output.write(json.dumps(event, sort_keys=True) + "\n")


def iter_pcm16_wav_frames(
    audio_path: str | Path,
    *,
    sample_rate: int = 16000,
    frame_duration_seconds: float = 0.08,
) -> Iterable[bytes]:
    frame_count = max(1, int(round(sample_rate * frame_duration_seconds)))
    with wave.open(str(audio_path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != sample_rate:
            raise ValueError(
                "true-streaming audio must be little-endian 16 kHz mono int16 PCM WAV"
            )
        while True:
            frame = wav.readframes(frame_count)
            if not frame:
                break
            yield frame


def build_vdo_true_streaming_runner(
    *,
    source: str,
    runner_url: Optional[str],
    output_dir: str,
    segment_count: int,
    segment_duration_seconds: float,
    startup_seconds: float,
    session_id: str,
    password: str,
    signaling_server: str = "",
    buffer_ms: int,
    audio_only: bool,
    language: str,
    preset: str,
    max_speakers: int,
    sample_rate: int,
    frame_duration_seconds: float,
    client_cls: Any = NemoTrueStreamingWebSocketClient,
    client_init_kwargs: Optional[Dict[str, Any]] = None,
) -> LivepeerTrueStreamingSessionRunner:
    artifact_dir = _true_streaming_artifact_dir(output_dir=output_dir, source=source)
    stream_source = LivepeerVDONinjaAudioSegmentSource(
        source=source,
        output_dir=artifact_dir,
        segment_duration_seconds=segment_duration_seconds,
        startup_seconds=startup_seconds,
        password=password,
        signaling_server=signaling_server,
        buffer_ms=buffer_ms,
        audio_only=audio_only,
        max_segments=segment_count,
    )
    config = LivepeerTrueStreamingSessionConfig(
        session_id=session_id or LivepeerTrueStreamingSessionConfig().session_id,
        language=language,
        preset=preset,
        max_speakers=max_speakers,
        sample_rate=sample_rate,
        frame_duration_seconds=frame_duration_seconds,
        artifact_root=str(artifact_dir),
    )
    return LivepeerTrueStreamingSessionRunner(
        stream_source=stream_source,
        client=_build_transport_client(
            client_cls=client_cls,
            base_url=runner_url,
            client_init_kwargs=client_init_kwargs,
        ),
        session_config=config,
        artifact_dir=artifact_dir,
    )


def build_vdo_direct_true_streaming_runner(
    *,
    source: str,
    runner_url: Optional[str],
    output_dir: str,
    duration_seconds: float,
    startup_timeout_seconds: float,
    session_id: str,
    password: str,
    signaling_server: str = "",
    buffer_ms: int,
    language: str,
    preset: str,
    max_speakers: int,
    sample_rate: int,
    frame_duration_seconds: float,
    client_cls: Any = NemoTrueStreamingWebSocketClient,
    client_init_kwargs: Optional[Dict[str, Any]] = None,
) -> LivepeerVDONinjaDirectTrueStreamingRunner:
    resolved = resolve_vdo_stream_source(
        source=source,
        signaling_server=signaling_server or init_vdo_signaling_server_url(),
        password=password,
        timeout_seconds=startup_timeout_seconds,
    )
    resolved_source = str(resolved["source"])
    resolved_signaling_server = str(resolved.get("signaling_server") or "")
    resolved_password = str(resolved.get("password", password))
    artifact_dir = _direct_true_streaming_artifact_dir(output_dir=output_dir, source=resolved_source)
    config = LivepeerTrueStreamingSessionConfig(
        session_id=session_id or LivepeerTrueStreamingSessionConfig().session_id,
        language=language,
        preset=preset,
        max_speakers=max_speakers,
        sample_rate=sample_rate,
        frame_duration_seconds=frame_duration_seconds,
        artifact_root=str(artifact_dir),
    )
    return LivepeerVDONinjaDirectTrueStreamingRunner(
        source=resolved_source,
        client=_build_transport_client(
            client_cls=client_cls,
            base_url=runner_url,
            client_init_kwargs=client_init_kwargs,
            initial_receive_timeout_seconds=startup_timeout_seconds,
        ),
        session_config=config,
        artifact_dir=artifact_dir,
        duration_seconds=duration_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
        password=resolved_password,
        signaling_server=resolved_signaling_server,
        buffer_ms=buffer_ms,
    )


def build_local_audio_ingest_true_streaming_runner(
    *,
    source: str,
    runner_url: Optional[str],
    local_audio_ingest_url: Optional[str],
    output_dir: str,
    duration_seconds: float,
    startup_timeout_seconds: float,
    session_id: str,
    language: str,
    preset: str,
    max_speakers: int,
    sample_rate: int,
    frame_duration_seconds: float,
    client_cls: Any = NemoTrueStreamingWebSocketClient,
    ingest_client_cls: Any = LocalAudioIngestWebSocketClient,
    client_init_kwargs: Optional[Dict[str, Any]] = None,
) -> LivepeerLocalAudioIngressTrueStreamingRunner:
    artifact_dir = _local_audio_ingest_artifact_dir(output_dir=output_dir, source=source)
    config = LivepeerTrueStreamingSessionConfig(
        session_id=session_id or LivepeerTrueStreamingSessionConfig().session_id,
        language=language,
        preset=preset,
        max_speakers=max_speakers,
        sample_rate=sample_rate,
        frame_duration_seconds=frame_duration_seconds,
        artifact_root=str(artifact_dir),
    )
    return LivepeerLocalAudioIngressTrueStreamingRunner(
        source=source,
        client=_build_transport_client(
            client_cls=client_cls,
            base_url=runner_url,
            client_init_kwargs=client_init_kwargs,
            initial_receive_timeout_seconds=startup_timeout_seconds,
        ),
        session_config=config,
        artifact_dir=artifact_dir,
        duration_seconds=duration_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
        ingest_client=ingest_client_cls(base_url=local_audio_ingest_url),
    )


def _true_streaming_ws_url(
    base_url: str,
    *,
    session_id: str,
    language: str,
    preset: str,
    max_speakers: int,
    sample_rate: int,
) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = (parsed.path.rstrip("/") if parsed.path else "") + "/v1/audio/transcriptions/stream"
    query = urlencode(
        {
            "session_id": session_id,
            "language": language,
            "preset": preset,
            "max_speakers": max_speakers,
            "sample_rate": sample_rate,
        }
    )
    return urlunparse((scheme, parsed.netloc, path, "", query, ""))


def _true_streaming_artifact_dir(*, output_dir: str, source: str) -> Path:
    stream_id = parse_vdo_stream_id(source)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path(output_dir) / f"vdo-{stream_id.replace('stream_', '')}-true-streaming-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _direct_true_streaming_artifact_dir(*, output_dir: str, source: str) -> Path:
    stream_id = safe_file_prefix(parse_vdo_stream_id(source)).replace("stream_", "")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path(output_dir) / f"vdo-{stream_id}-direct-true-streaming-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _local_audio_ingest_artifact_dir(*, output_dir: str, source: str) -> Path:
    session_id = safe_file_prefix(parse_local_audio_ingest_source(source=source)["session_id"])
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path(output_dir) / f"local-ingest-{session_id}-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _true_streaming_result_payload(
    *,
    session_id: str,
    stream_id: str,
    captured_segments: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    events_path: Path,
    result_path: Path,
    transcript_text_path: Path,
) -> Dict[str, Any]:
    resolved_session_id = session_id
    for event in reversed(events):
        event_session_id = str(event.get("session_id") or "").strip()
        if event_session_id:
            resolved_session_id = event_session_id
            break
    transcript_events = [
        event
        for event in events
        if event.get("event_type") in {"transcript.segment", "speaker.update"}
    ]
    transcript_segments = [
        event
        for event in events
        if event.get("event_type") == "transcript.segment"
        and str(event.get("text") or "").strip()
    ]
    text_segments = _selected_transcript_segments(transcript_segments)
    text = " ".join(str(event.get("text") or "").strip() for event in text_segments).strip()
    speakers = _speaker_summaries_from_events(events)
    return {
        "session_id": resolved_session_id,
        "stream_id": stream_id,
        "status": "closed" if any(event.get("event_type") == "transcript.session.finished" for event in events) else "active",
        "captured_segments": captured_segments,
        "audio_paths": [
            str(segment.get("audio_path"))
            for segment in captured_segments
            if segment.get("audio_path")
        ],
        "text": text,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "transcript_events": transcript_events,
        "transcript_event_count": len(transcript_events),
        "events": events,
        "event_count": len(events),
        "events_jsonl_path": str(events_path),
        "result_json_path": str(result_path),
        "transcript_text_path": str(transcript_text_path),
    }


def _selected_transcript_segments(transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not transcript_segments:
        return []
    consolidated = _consolidate_transcript_segments(transcript_segments)
    final_segments = [event for event in consolidated if event.get("is_final") is True]
    if not final_segments:
        return consolidated
    final_coverage = _transcript_coverage_seconds(final_segments)
    total_coverage = _transcript_coverage_seconds(consolidated)
    # Some realtime runners emit provisional segments throughout the session
    # and only one tiny final segment on close. In that case, preferring
    # finals-only collapses the transcript to the trailing word(s).
    if total_coverage > 0 and final_coverage >= total_coverage * 0.8:
        return final_segments
    return consolidated


def _consolidate_transcript_segments(
    transcript_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    latest_by_key: Dict[tuple[str, float, float], Dict[str, Any]] = {}
    for event in transcript_segments:
        key = (
            str(event.get("speaker") or ""),
            _event_seconds(event.get("start", 0.0)),
            _event_seconds(event.get("end", 0.0)),
        )
        current = latest_by_key.get(key)
        if current is None:
            latest_by_key[key] = event
            continue
        if event.get("is_final") is True and current.get("is_final") is not True:
            latest_by_key[key] = event
            continue
        latest_by_key[key] = event
    return sorted(
        latest_by_key.values(),
        key=lambda event: (
            _event_seconds(event.get("start", 0.0)),
            _event_seconds(event.get("end", 0.0)),
            str(event.get("speaker") or ""),
        ),
    )


def _transcript_coverage_seconds(transcript_segments: List[Dict[str, Any]]) -> float:
    covered = 0.0
    for event in transcript_segments:
        start = _event_seconds(event.get("start", 0.0))
        end = _event_seconds(event.get("end", start))
        covered += max(0.0, end - start)
    return round(covered, 3)


def _speaker_summaries_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    totals: Dict[str, float] = {}
    for event in events:
        speaker = event.get("speaker")
        if not speaker:
            continue
        start = _event_seconds(event.get("start", 0.0))
        end = _event_seconds(event.get("end", start))
        totals[str(speaker)] = totals.get(str(speaker), 0.0) + max(0.0, end - start)
    return [
        {"id": speaker, "talk_seconds": round(seconds, 3)}
        for speaker, seconds in sorted(totals.items())
    ]


def _event_seconds(value: Any) -> float:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return 0.0


def _json_event(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_websocket_connect() -> Any:
    try:
        from websockets.sync.client import connect
    except ImportError as error:  # pragma: no cover - exercised in minimal environments
        raise RuntimeError(
            "websockets is required for true-streaming runner WebSocket sessions"
        ) from error
    return connect


def _is_websocket_closed_error(error: BaseException) -> bool:
    closed_error_names = {cls.__name__ for cls in type(error).__mro__}
    if any(name.startswith("ConnectionClosed") for name in closed_error_names):
        return True
    if isinstance(error, (BrokenPipeError, ConnectionResetError, EOFError, StopIteration)):
        return True
    message = str(error).lower()
    return (
        "connection closed" in message
        or "websocket closed" in message
        or "no close frame received or sent" in message
    )


def _connect_without_client_pings(connect: Any, url: str) -> Any:
    try:
        return connect(url, ping_interval=None)
    except TypeError:
        return connect(url)
