"""Additive streaming session runtime for Livepeer-backed Roboflow workflows."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Protocol

from .ingest import DEFAULT_INGEST_OUTPUT_DIR, DEFAULT_RASPBERRY_NINJA, RollingSegment, record_vdo_segment


BackpressurePolicy = Literal["block", "drop_oldest", "drop_newest"]


@dataclass(frozen=True)
class LivepeerStreamingSessionConfig:
    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    backpressure_policy: BackpressurePolicy = "block"
    max_queue_size: int = 64
    audio_chunk_seconds: float = 30.0
    audio_chunk_overlap_seconds: float = 0.0
    max_segment_latency_seconds: Optional[float] = None
    artifact_root: str = str(DEFAULT_INGEST_OUTPUT_DIR)
    emit_billing_events: bool = True
    stop_on_step_error: bool = True
    source_reconnect_policy: str = "none"
    runner_mode: str = "livepeer-vdo-audio"

    def __post_init__(self) -> None:
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.audio_chunk_seconds <= 0:
            raise ValueError("audio_chunk_seconds must be positive")
        if self.audio_chunk_overlap_seconds < 0:
            raise ValueError("audio_chunk_overlap_seconds must not be negative")


@dataclass(frozen=True)
class LivepeerStreamingEvent:
    event_type: str
    session_id: str
    stream_id: Optional[str] = None
    segment_id: Optional[str] = None
    segment_index: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class AudioSegmentSource(Protocol):
    stream_id: str

    def open(self) -> None:
        ...

    def segments(self) -> Iterable[RollingSegment]:
        ...

    def stop(self) -> None:
        ...

    def close(self) -> None:
        ...


class LivepeerVDONinjaAudioSegmentSource:
    """VDO.Ninja/Raspberry.Ninja source adapter that emits bounded audio windows."""

    def __init__(
        self,
        *,
        source: str,
        output_dir: str | Path = DEFAULT_INGEST_OUTPUT_DIR,
        segment_duration_seconds: float = 30.0,
        startup_seconds: float = 8.0,
        password: str = "",
        signaling_server: str = "",
        buffer_ms: int = 300,
        audio_only: bool = True,
        max_segments: Optional[int] = None,
        raspberry_ninja_path: str | Path = DEFAULT_RASPBERRY_NINJA,
    ) -> None:
        if max_segments is not None and max_segments <= 0:
            raise ValueError("max_segments must be positive when provided")
        self.source = source
        self.output_dir = output_dir
        self.segment_duration_seconds = segment_duration_seconds
        self.startup_seconds = startup_seconds
        self.password = password
        self.signaling_server = signaling_server
        self.buffer_ms = buffer_ms
        self.audio_only = audio_only
        self.max_segments = max_segments
        self.raspberry_ninja_path = raspberry_ninja_path
        self.stream_id = source
        self._stop_requested = threading.Event()

    def open(self) -> None:
        from .ingest import parse_vdo_stream_id

        self.stream_id = parse_vdo_stream_id(self.source)

    def segments(self) -> Iterator[RollingSegment]:
        index = 0
        while not self._stop_requested.is_set():
            if self.max_segments is not None and index >= self.max_segments:
                break
            yield record_vdo_segment(
                source=self.source,
                output_dir=self.output_dir,
                duration_seconds=self.segment_duration_seconds,
                startup_seconds=self.startup_seconds if index == 0 else 0.0,
                password=self.password,
                signaling_server=self.signaling_server,
                buffer_ms=self.buffer_ms,
                audio_only=self.audio_only,
                segment_index=index,
                raspberry_ninja_path=self.raspberry_ninja_path,
            )
            index += 1

    def stop(self) -> None:
        self._stop_requested.set()

    def close(self) -> None:
        self.stop()


class LivepeerStreamingSessionRunner:
    """Compile once, run a normal Roboflow workflow for each source audio segment."""

    def __init__(
        self,
        *,
        workflow_definition: Dict[str, Any],
        init_parameters: Optional[Dict[str, Any]],
        stream_source: AudioSegmentSource,
        session_config: LivepeerStreamingSessionConfig,
        workflow_runtime_parameters: Optional[Dict[str, Any]] = None,
        execution_engine: Any = None,
    ) -> None:
        self.workflow_definition = workflow_definition
        self.init_parameters = init_parameters or {}
        self.stream_source = stream_source
        self.session_config = session_config
        self.workflow_runtime_parameters = workflow_runtime_parameters or {}
        self._execution_engine = execution_engine
        self._events: queue.Queue[LivepeerStreamingEvent] = queue.Queue(
            maxsize=session_config.max_queue_size
        )
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._ended = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def init(
        cls,
        *,
        workflow_definition: Dict[str, Any],
        init_parameters: Optional[Dict[str, Any]] = None,
        stream_source: AudioSegmentSource,
        session_config: Optional[LivepeerStreamingSessionConfig] = None,
        workflow_runtime_parameters: Optional[Dict[str, Any]] = None,
    ) -> "LivepeerStreamingSessionRunner":
        config = session_config or LivepeerStreamingSessionConfig()
        return cls(
            workflow_definition=workflow_definition,
            init_parameters=init_parameters,
            stream_source=stream_source,
            session_config=config,
            workflow_runtime_parameters=workflow_runtime_parameters,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"livepeer-streaming-session-{self.session_config.session_id}",
            daemon=True,
        )
        self._thread.start()

    def pause(self) -> None:
        self._pause_requested.set()

    def resume(self) -> None:
        self._pause_requested.clear()

    def stop(self) -> None:
        self._stop_requested.set()
        self.stream_source.stop()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def events(self) -> Iterator[LivepeerStreamingEvent]:
        while not self._ended.is_set() or not self._events.empty():
            try:
                yield self._events.get(timeout=0.1)
            except queue.Empty:
                continue

    def _run_loop(self) -> None:
        try:
            self._emit("session.started", payload={"config": self._config_payload()})
            engine = self._execution_engine or self._build_execution_engine()
            self.stream_source.open()
            self._emit("source.connected", stream_id=self.stream_source.stream_id)
            for segment in self.stream_source.segments():
                if self._stop_requested.is_set():
                    break
                while self._pause_requested.is_set() and not self._stop_requested.is_set():
                    time.sleep(0.05)
                if self._stop_requested.is_set():
                    break
                self._process_segment(engine=engine, segment=segment)
        except Exception as error:
            self._emit_error(error)
        finally:
            try:
                self.stream_source.close()
            finally:
                self._emit("session.ended", payload={"stopped": self._stop_requested.is_set()})
                self._ended.set()

    def _build_execution_engine(self) -> Any:
        from inference.core.workflows.execution_engine.core import ExecutionEngine

        return ExecutionEngine.init(
            workflow_definition=self.workflow_definition,
            init_parameters=self.init_parameters,
            max_concurrent_steps=1,
        )

    def _process_segment(self, *, engine: Any, segment: RollingSegment) -> None:
        segment_id = f"{segment.stream_id}:{segment.index}"
        segment_payload = segment.as_dict()
        self._emit(
            "source.audio_chunk",
            stream_id=segment.stream_id,
            segment_id=segment_id,
            segment_index=segment.index,
            payload=segment_payload,
        )
        for artifact_name, artifact_path in (
            ("recording", segment.recording_path),
            ("audio", segment.audio_path),
            ("log", segment.log_path),
        ):
            self._emit(
                "artifact.created",
                stream_id=segment.stream_id,
                segment_id=segment_id,
                segment_index=segment.index,
                payload={"artifact_type": artifact_name, "path": str(artifact_path)},
            )

        runtime_parameters = {
            **self.workflow_runtime_parameters,
            "audio_path": str(segment.audio_path),
            "stream_id": segment.stream_id,
            "segment_id": segment_id,
            "segment_index": segment.index,
            "segment_started_at_epoch": segment.started_at_epoch,
            "segment_completed_at_epoch": segment.completed_at_epoch,
            "audio_duration_seconds": segment.audio_duration_seconds,
        }
        try:
            output = engine.run(runtime_parameters=runtime_parameters)
        except Exception as error:
            self._emit_error(
                error,
                stream_id=segment.stream_id,
                segment_id=segment_id,
                segment_index=segment.index,
            )
            if self.session_config.stop_on_step_error:
                self.stop()
            return

        self._emit(
            "workflow.output",
            stream_id=segment.stream_id,
            segment_id=segment_id,
            segment_index=segment.index,
            payload={"runtime_parameters": runtime_parameters, "output": output},
        )
        if self.session_config.emit_billing_events:
            units = self._extract_actual_units(output)
            if units is not None:
                self._emit(
                    "billing.usage",
                    stream_id=segment.stream_id,
                    segment_id=segment_id,
                    segment_index=segment.index,
                    payload={"actual_units": units},
                )

    def _extract_actual_units(self, output: Any) -> Optional[int]:
        if isinstance(output, list) and output and isinstance(output[0], dict):
            units = output[0].get("actual_units")
            if isinstance(units, int):
                return units
        if isinstance(output, dict):
            units = output.get("actual_units")
            if isinstance(units, int):
                return units
        return None

    def _emit_error(
        self,
        error: Exception,
        *,
        stream_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_index: Optional[int] = None,
    ) -> None:
        self._emit(
            "error",
            stream_id=stream_id,
            segment_id=segment_id,
            segment_index=segment_index,
            payload={"error_type": type(error).__name__, "message": str(error)},
        )

    def _emit(
        self,
        event_type: str,
        *,
        stream_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_index: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = LivepeerStreamingEvent(
            event_type=event_type,
            session_id=self.session_config.session_id,
            stream_id=stream_id,
            segment_id=segment_id,
            segment_index=segment_index,
            payload=payload or {},
        )
        if self.session_config.backpressure_policy == "block":
            self._events.put(event)
            return
        if self.session_config.backpressure_policy == "drop_newest":
            try:
                self._events.put_nowait(event)
            except queue.Full:
                return
            return
        if self.session_config.backpressure_policy == "drop_oldest":
            try:
                self._events.put_nowait(event)
            except queue.Full:
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    pass
                self._events.put_nowait(event)
            return
        raise ValueError(f"Unknown backpressure policy: {self.session_config.backpressure_policy}")

    def _config_payload(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_config.session_id,
            "backpressure_policy": self.session_config.backpressure_policy,
            "max_queue_size": self.session_config.max_queue_size,
            "audio_chunk_seconds": self.session_config.audio_chunk_seconds,
            "audio_chunk_overlap_seconds": self.session_config.audio_chunk_overlap_seconds,
            "max_segment_latency_seconds": self.session_config.max_segment_latency_seconds,
            "artifact_root": self.session_config.artifact_root,
            "emit_billing_events": self.session_config.emit_billing_events,
            "stop_on_step_error": self.session_config.stop_on_step_error,
            "source_reconnect_policy": self.session_config.source_reconnect_policy,
            "runner_mode": self.session_config.runner_mode,
        }
