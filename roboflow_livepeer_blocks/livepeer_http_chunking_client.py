"""Alternate LOC remote transport that batches PCM/audio files into HTTP chunk requests."""

from __future__ import annotations

import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audio import AudioChunk, materialize_audio_chunks
from .client import LivepeerOpenClearinghouseClient


def fallback_offering_for_streaming(offering: str) -> str:
    resolved = str(offering or "").strip()
    if resolved.endswith("-stream"):
        return resolved[: -len("-stream")]
    if resolved.endswith("_stream"):
        return resolved[: -len("_stream")]
    return resolved


class LivepeerRemoteHttpChunkingClient:
    """Fallback transport that uses LOC's working HTTP handoff path on bounded chunks."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        capability: str,
        offering: str,
        chunk_size_seconds: float = 10.0,
        http_client: Any = None,
    ) -> None:
        if chunk_size_seconds <= 0:
            raise ValueError("chunk_size_seconds must be positive")
        self.api_key = api_key
        self.base_url = base_url
        self.capability = capability
        self.offering = offering
        self.chunk_size_seconds = float(chunk_size_seconds)
        self._http_client = http_client

    def connect_session(
        self,
        *,
        session_id: str,
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        sample_rate: int = 16000,
        frame_duration_seconds: float = 0.08,
    ) -> "_LivepeerRemoteHttpChunkingSession":
        return _LivepeerRemoteHttpChunkingSession(
            api_key=self.api_key,
            base_url=self.base_url,
            capability=self.capability,
            offering=self.offering,
            chunk_size_seconds=self.chunk_size_seconds,
            session_id=session_id,
            language=language,
            preset=preset,
            max_speakers=max_speakers,
            sample_rate=sample_rate,
            frame_duration_seconds=frame_duration_seconds,
            http_client=self._http_client,
        )


class _LivepeerRemoteHttpChunkingSession:
    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        capability: str,
        offering: str,
        chunk_size_seconds: float,
        session_id: str,
        language: str,
        preset: str,
        max_speakers: int,
        sample_rate: int,
        frame_duration_seconds: float,
        http_client: Any = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.capability = capability
        self.offering = offering
        self.chunk_size_seconds = chunk_size_seconds
        self.session_id = session_id
        self.language = language
        self.preset = preset
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate
        self.frame_duration_seconds = frame_duration_seconds
        self.events: List[Dict[str, Any]] = []
        self._http_client = http_client
        self._client: Optional[LivepeerOpenClearinghouseClient] = None
        self._buffer = bytearray()
        self._chunk_index = 0
        self._consumed_audio_seconds = 0.0
        self._finished = False
        self._chunk_dir: Optional[tempfile.TemporaryDirectory[str]] = None

    def __enter__(self) -> "_LivepeerRemoteHttpChunkingSession":
        self._client = LivepeerOpenClearinghouseClient(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self._http_client,
        )
        self._chunk_dir = tempfile.TemporaryDirectory(prefix="livepeer-remote-http-chunks-")
        self.events.append(
            {
                "event_type": "session.snapshot",
                "session_id": self.session_id,
                "status": "active",
                "transport": "livepeer_remote_http_chunking",
                "capability": self.capability,
                "offering": self.offering,
                "chunk_size_seconds": self.chunk_size_seconds,
                "language": self.language,
                "preset": self.preset,
                "max_speakers": self.max_speakers,
                "sample_rate": self.sample_rate,
                "duration_seconds": 0.0,
            }
        )
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            if not self._finished:
                self.finish()
        finally:
            if self._chunk_dir is not None:
                self._chunk_dir.cleanup()
            if self._client is not None:
                self._client.close()

    def send_audio_file(
        self,
        audio_path: str | Path,
        *,
        source_segment: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._client is None or self._chunk_dir is None:
            raise RuntimeError("HTTP chunking session must be entered before use")
        start_index = len(self.events)
        chunks = materialize_audio_chunks(
            audio_path=audio_path,
            output_dir=self._chunk_dir.name,
            chunk_size_seconds=self.chunk_size_seconds,
        )
        emitted: List[Dict[str, Any]] = []
        for source_chunk in chunks:
            start_seconds = self._consumed_audio_seconds
            end_seconds = start_seconds + source_chunk.duration_seconds
            chunk = AudioChunk(
                index=self._chunk_index,
                path=source_chunk.path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=source_chunk.duration_seconds,
                temporary=source_chunk.temporary,
            )
            self._chunk_index += 1
            self._consumed_audio_seconds = end_seconds
            emitted.extend(
                self._transcribe_chunk(
                    chunk=chunk,
                    source_segment=source_segment,
                )
            )
        return emitted or self.events[start_index:]

    def send_audio_frame(
        self,
        frame: bytes,
        *,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._client is None or self._chunk_dir is None:
            raise RuntimeError("HTTP chunking session must be entered before use")
        self._buffer.extend(frame)
        minimum_bytes = max(2, int(self.chunk_size_seconds * self.sample_rate) * 2)
        if len(self._buffer) < minimum_bytes:
            return []
        return self._flush_buffer(force=False, source_event=source_event)

    def finish(self) -> List[Dict[str, Any]]:
        if self._finished:
            return []
        self._finished = True
        start_index = len(self.events)
        if self._buffer:
            self._flush_buffer(force=True, source_event=None)
        self.events.append(
            {
                "event_type": "transcript.session.finished",
                "session_id": self.session_id,
                "status": "closed",
                "finish_reason": "http_chunking_complete",
                "duration_seconds": round(self._consumed_audio_seconds, 6),
                "is_final": True,
            }
        )
        return self.events[start_index:]

    def _flush_buffer(
        self,
        *,
        force: bool,
        source_event: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        chunk_bytes = max(2, int(self.chunk_size_seconds * self.sample_rate) * 2)
        if not force and len(self._buffer) < chunk_bytes:
            return []
        if force:
            payload = bytes(self._buffer)
            self._buffer.clear()
        else:
            payload = bytes(self._buffer[:chunk_bytes])
            del self._buffer[:chunk_bytes]
        duration_seconds = len(payload) / 2 / self.sample_rate
        if duration_seconds <= 0:
            return []
        chunk = self._write_pcm_chunk(
            pcm_bytes=payload,
            duration_seconds=duration_seconds,
        )
        return self._transcribe_chunk(
            chunk=chunk,
            source_event=source_event,
        )

    def _write_pcm_chunk(
        self,
        *,
        pcm_bytes: bytes,
        duration_seconds: float,
    ) -> AudioChunk:
        if self._chunk_dir is None:
            raise RuntimeError("HTTP chunking session temporary directory missing")
        start_seconds = self._consumed_audio_seconds
        end_seconds = start_seconds + duration_seconds
        chunk_path = Path(self._chunk_dir.name) / f"http-chunk-{self._chunk_index:04d}.wav"
        with wave.open(str(chunk_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm_bytes)
        chunk = AudioChunk(
            index=self._chunk_index,
            path=chunk_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=duration_seconds,
            temporary=True,
        )
        self._chunk_index += 1
        self._consumed_audio_seconds = end_seconds
        return chunk

    def _transcribe_chunk(
        self,
        *,
        chunk: AudioChunk,
        source_segment: Optional[Dict[str, Any]] = None,
        source_event: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("HTTP chunking client unavailable")
        start_index = len(self.events)
        try:
            result = self._client.transcribe_chunk(
                chunk,
                capability=self.capability,
                offering=self.offering,
                response_format="json",
            )
        except RuntimeError as error:
            if not _is_silence_only_chunk_error(error):
                raise
            skipped_event = {
                "event_type": "transcript.chunk.skipped",
                "session_id": self.session_id,
                "skip_reason": "silence",
                "text_status": "silence",
                "start": chunk.start_seconds,
                "end": chunk.end_seconds,
                "duration_seconds": chunk.duration_seconds,
                "is_final": True,
                "actual_units": 0,
                "transport": "livepeer_remote_http_chunking",
                "error_type": error.__class__.__name__,
                "error": str(error),
            }
            if source_segment:
                skipped_event["source_segment"] = source_segment
            if source_event:
                skipped_event["source_event"] = source_event
            self.events.append(skipped_event)
            return self.events[start_index:]
        segment_event = {
            "event_type": "transcript.segment",
            "session_id": self.session_id,
            "text": result.text,
            "start": chunk.start_seconds,
            "end": chunk.end_seconds,
            "duration_seconds": chunk.duration_seconds,
            "is_final": True,
            "speaker": None,
            "actual_units": result.actual_units,
            "job_id": result.job_id,
            "work_id": result.work_id,
            "transport": "livepeer_remote_http_chunking",
        }
        if source_segment:
            segment_event["source_segment"] = source_segment
        if source_event:
            segment_event["source_event"] = source_event
        self.events.append(segment_event)
        self.events.append(
            {
                "event_type": "payment.job.completed",
                "session_id": self.session_id,
                "job_id": result.job_id,
                "work_id": result.work_id,
                "actual_units": result.actual_units,
                "offering": self.offering,
                "transport": "livepeer_remote_http_chunking",
            }
        )
        return self.events[start_index:]


class LivepeerRemoteFallbackTransportClient:
    """Prefer brokered ws-realtime, but drop to HTTP chunking when the WS path is unavailable."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str,
        capability: str,
        realtime_offering: str,
        estimated_runway_units: Optional[int] = None,
        max_total_units: Optional[int] = None,
        chunk_size_seconds: float = 10.0,
        websocket_connect: Any = None,
        realtime_http_client: Any = None,
        chunk_http_client: Any = None,
        receive_timeout_seconds: float = 0.05,
        initial_receive_timeout_seconds: Optional[float] = None,
        finish_receive_timeout_seconds: float = 5.0,
    ) -> None:
        from .livepeer_realtime_client import LivepeerRemoteTrueStreamingWebSocketClient

        fallback_offering = fallback_offering_for_streaming(realtime_offering)
        self._primary = LivepeerRemoteTrueStreamingWebSocketClient(
            api_key=api_key,
            base_url=base_url,
            capability=capability,
            offering=realtime_offering,
            estimated_runway_units=estimated_runway_units,
            max_total_units=max_total_units,
            websocket_connect=websocket_connect,
            http_client=realtime_http_client,
            receive_timeout_seconds=receive_timeout_seconds,
            initial_receive_timeout_seconds=initial_receive_timeout_seconds,
            finish_receive_timeout_seconds=finish_receive_timeout_seconds,
        )
        self._fallback = LivepeerRemoteHttpChunkingClient(
            api_key=api_key,
            base_url=base_url,
            capability=capability,
            offering=fallback_offering,
            chunk_size_seconds=chunk_size_seconds,
            http_client=chunk_http_client,
        )

    def connect_session(self, **kwargs: Any) -> "_LivepeerRemoteFallbackTransportSession":
        return _LivepeerRemoteFallbackTransportSession(
            primary_session=self._primary.connect_session(**kwargs),
            fallback_client=self._fallback,
            session_kwargs=kwargs,
        )


class _LivepeerRemoteFallbackTransportSession:
    def __init__(
        self,
        *,
        primary_session: Any,
        fallback_client: LivepeerRemoteHttpChunkingClient,
        session_kwargs: Dict[str, Any],
    ) -> None:
        self._primary_session = primary_session
        self._fallback_client = fallback_client
        self._session_kwargs = dict(session_kwargs)
        self._active_session: Any = None
        self._fallback_prefix_events: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def __enter__(self) -> "_LivepeerRemoteFallbackTransportSession":
        try:
            self._active_session = self._primary_session.__enter__()
            self.events = self._active_session.events
            return self
        except RuntimeError as error:
            if not _has_handshake_failure_event(self._primary_session):
                raise
            primary_events = list(getattr(self._primary_session, "events", []))
            close_event = self._close_failed_primary_session()
            self._fallback_prefix_events = primary_events + [
                close_event,
                {
                    "event_type": "transcript.transport.fallback",
                    "session_id": str(self._session_kwargs.get("session_id") or ""),
                    "from_transport": "livepeer_remote_ws_realtime",
                    "to_transport": "livepeer_remote_http_chunking",
                    "reason_type": error.__class__.__name__,
                    "reason": str(error),
                }
            ]
            fallback_session = self._fallback_client.connect_session(**self._session_kwargs)
            self._active_session = fallback_session.__enter__()
            self.events = self._fallback_prefix_events + list(self._active_session.events)
            return self

    def __exit__(self, *args: Any) -> None:
        if self._active_session is None:
            return None
        return self._active_session.__exit__(*args)

    def send_audio_file(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        emitted = self._active_session.send_audio_file(*args, **kwargs)
        self._sync_events()
        return emitted

    def send_audio_frame(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        emitted = self._active_session.send_audio_frame(*args, **kwargs)
        self._sync_events()
        return emitted

    def finish(self) -> List[Dict[str, Any]]:
        emitted = self._active_session.finish()
        self._sync_events()
        return emitted

    def _sync_events(self) -> None:
        if self._active_session is None:
            return
        if self._fallback_prefix_events:
            self.events = self._fallback_prefix_events + list(self._active_session.events)
        else:
            self.events = self._active_session.events

    def _close_failed_primary_session(self) -> Dict[str, Any]:
        close_endpoint = str(getattr(self._primary_session, "session_info", {}).get("close_endpoint") or "")
        if not close_endpoint:
            return {
                "event_type": "payment.session.close_skipped",
                "session_id": str(self._session_kwargs.get("session_id") or ""),
                "reason": "missing_close_endpoint_after_ws_failure",
            }
        client = getattr(self._primary_session, "_client", None)
        base_url = getattr(self._primary_session, "base_url", "")
        api_key = getattr(self._primary_session, "api_key", None)
        if client is None or not base_url or not api_key:
            return {
                "event_type": "payment.session.close_skipped",
                "session_id": str(self._session_kwargs.get("session_id") or ""),
                "reason": "missing_primary_client_context_after_ws_failure",
            }
        try:
            response = client.post(
                f"{base_url.rstrip('/')}/{close_endpoint.lstrip('/')}",
                headers={
                    "X-API-Key": api_key,
                    "Livepeer-Open-Clearinghouse-SDK": "roboflow-livepeer-blocks-realtime/0.1.0",
                },
                json={
                    "actual_units": 0,
                    "outcome": "ws_handshake_failed_fallback_to_http_chunking",
                },
            )
            response.raise_for_status()
        except Exception as error:
            return {
                "event_type": "payment.session.close_failed",
                "session_id": str(self._session_kwargs.get("session_id") or ""),
                "error_type": error.__class__.__name__,
                "error": str(error),
                "close_reason": "ws_handshake_failed_fallback_to_http_chunking",
            }
        return {
            "event_type": "payment.session.closed",
            "billing_session_id": getattr(self._primary_session, "session_info", {}).get("session_id", ""),
            "work_id": getattr(self._primary_session, "session_info", {}).get("work_id", ""),
            "actual_units": 0,
            "settlement": response.json(),
            "close_reason": "ws_handshake_failed_fallback_to_http_chunking",
        }


def _has_handshake_failure_event(session: Any) -> bool:
    return any(
        isinstance(event, dict)
        and event.get("event_type") == "livepeer.realtime.websocket.handshake.failed"
        for event in getattr(session, "events", [])
    )


def _is_silence_only_chunk_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        "broker request failed with status 500" in message
        and "server_error" in message
        and "silence" in message
    )
