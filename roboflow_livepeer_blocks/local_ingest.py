"""Localhost audio ingest service and client helpers for extension capture."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode, urlparse, urlunparse

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from .config import init_local_audio_ingest_url


LOCAL_AUDIO_FRAME_SENTINEL = object()


class LocalAudioIngestSourceInfo(BaseModel):
    session_id: str = Field()
    source: str = Field()
    source_type: str = Field(default="localhost_ingest")
    ingest_url: str = Field()
    consume_url: str = Field()
    status_url: str = Field()
    sample_format: str = Field(default="s16le")
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)


@dataclass
class _LocalAudioIngestSession:
    session_id: str
    ingest_url: str
    consume_url: str
    status_url: str
    sample_rate: int
    channels: int
    sample_format: str
    source: str
    source_label: str
    created_at_epoch: float = field(default_factory=time.time)
    producer_connected_at_epoch: Optional[float] = None
    producer_closed_at_epoch: Optional[float] = None
    last_frame_at_epoch: Optional[float] = None
    total_audio_bytes: int = 0
    total_frames: int = 0
    producer_connection_id: Optional[str] = None
    consumer_count: int = 0
    consumer_queues: Dict[str, asyncio.Queue[object]] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return (
            self.producer_connected_at_epoch is not None
            and self.producer_closed_at_epoch is None
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "source_type": "localhost_ingest",
            "ingest_url": self.ingest_url,
            "consume_url": self.consume_url,
            "status_url": self.status_url,
            "sample_format": self.sample_format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "source_label": self.source_label,
            "created_at_epoch": self.created_at_epoch,
            "producer_connected_at_epoch": self.producer_connected_at_epoch,
            "producer_closed_at_epoch": self.producer_closed_at_epoch,
            "last_frame_at_epoch": self.last_frame_at_epoch,
            "total_audio_bytes": self.total_audio_bytes,
            "total_frames": self.total_frames,
            "consumer_count": self.consumer_count,
            "status": "open" if self.is_open else "closed",
        }


class LocalAudioIngestSessionManager:
    """In-memory producer/consumer session store for localhost audio ingest."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _LocalAudioIngestSession] = {}
        self._lock = asyncio.Lock()

    async def open_publisher(
        self,
        *,
        session_id: str,
        source: str,
        source_label: str,
        sample_rate: int,
        channels: int,
        sample_format: str,
    ) -> Dict[str, Any]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels != 1:
            raise ValueError("only mono audio is supported")
        if sample_format != "s16le":
            raise ValueError("only s16le PCM is supported")
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing and existing.is_open:
                raise ValueError(f"session {session_id!r} already has an active producer")
            info = parse_local_audio_ingest_source(
                source=source,
                default_base_url=init_local_audio_ingest_url(),
            )
            if existing is None:
                session = _LocalAudioIngestSession(
                    session_id=session_id,
                    ingest_url=info["ingest_url"],
                    consume_url=info["consume_url"],
                    status_url=info["status_url"],
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_format=sample_format,
                    source=source,
                    source_label=source_label,
                    producer_connected_at_epoch=time.time(),
                    producer_connection_id=f"producer_{uuid.uuid4().hex[:12]}",
                )
            else:
                session = existing
                session.ingest_url = info["ingest_url"]
                session.consume_url = info["consume_url"]
                session.status_url = info["status_url"]
                session.sample_rate = sample_rate
                session.channels = channels
                session.sample_format = sample_format
                session.source = source
                session.source_label = source_label
                session.producer_connected_at_epoch = time.time()
                session.producer_closed_at_epoch = None
                session.producer_connection_id = f"producer_{uuid.uuid4().hex[:12]}"
                session.last_frame_at_epoch = None
                session.total_audio_bytes = 0
                session.total_frames = 0
            self._sessions[session_id] = session
            return session.as_dict()

    async def publish_audio(self, *, session_id: str, frame: bytes) -> Dict[str, Any]:
        if not frame:
            raise ValueError("audio frame must not be empty")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"session {session_id!r} was not opened")
            session.total_audio_bytes += len(frame)
            session.total_frames += 1
            session.last_frame_at_epoch = time.time()
            queues = list(session.consumer_queues.values())
        for queue in queues:
            await queue.put(frame)
        return session.as_dict()

    async def close_publisher(self, *, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"session {session_id!r} was not opened")
            if session.producer_closed_at_epoch is None:
                session.producer_closed_at_epoch = time.time()
            queues = list(session.consumer_queues.values())
        for queue in queues:
            await queue.put(LOCAL_AUDIO_FRAME_SENTINEL)
        return session.as_dict()

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return session.as_dict()

    async def open_consumer(
        self,
        *,
        session_id: str,
        source: Optional[str] = None,
    ) -> tuple[str, asyncio.Queue[object], Dict[str, Any]]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                info = parse_local_audio_ingest_source(
                    source=source or session_id,
                    default_base_url=init_local_audio_ingest_url(),
                )
                session = _LocalAudioIngestSession(
                    session_id=session_id,
                    ingest_url=info["ingest_url"],
                    consume_url=info["consume_url"],
                    status_url=info["status_url"],
                    sample_rate=16000,
                    channels=1,
                    sample_format="s16le",
                    source=source or info["ingest_url"],
                    source_label="placeholder",
                )
                self._sessions[session_id] = session
            consumer_id = f"consumer_{uuid.uuid4().hex[:12]}"
            queue: asyncio.Queue[object] = asyncio.Queue()
            session.consumer_queues[consumer_id] = queue
            session.consumer_count = len(session.consumer_queues)
            snapshot = session.as_dict()
        return consumer_id, queue, snapshot

    async def close_consumer(self, *, session_id: str, consumer_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.consumer_queues.pop(consumer_id, None)
            session.consumer_count = len(session.consumer_queues)


def parse_local_audio_ingest_source(
    *,
    source: str,
    default_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    raw_source = source.strip()
    if not raw_source:
        raise ValueError("source must contain a local ingest session URL or session id")
    base_url = default_base_url or init_local_audio_ingest_url()
    if "://" not in raw_source:
        raw_source = _ingest_publish_url(base_url=base_url, session_id=raw_source)
    parsed = urlparse(raw_source)
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 4 or path_parts[:3] != ["v1", "ingest", "audio"]:
        raise ValueError(
            "source must target /v1/ingest/audio/{session_id} on the local ingest service"
        )
    session_id = path_parts[3]
    if not session_id:
        raise ValueError("local ingest session id must not be empty")
    status_url = urlunparse(
        (
            "http" if parsed.scheme == "ws" else "https" if parsed.scheme == "wss" else parsed.scheme,
            parsed.netloc,
            f"/v1/ingest/audio/{session_id}",
            "",
            "",
            "",
        )
    )
    ingest_url = urlunparse(
        (
            "ws" if parsed.scheme in {"http", "ws"} else "wss",
            parsed.netloc,
            f"/v1/ingest/audio/{session_id}",
            "",
            "",
            "",
        )
    )
    consume_url = urlunparse(
        (
            "ws" if parsed.scheme in {"http", "ws"} else "wss",
            parsed.netloc,
            f"/v1/ingest/audio/{session_id}/consume",
            "",
            "",
            "",
        )
    )
    return {
        "session_id": session_id,
        "source": raw_source,
        "status_url": status_url,
        "ingest_url": ingest_url,
        "consume_url": consume_url,
    }


def _ingest_publish_url(*, base_url: str, session_id: str) -> str:
    parsed = urlparse(base_url)
    scheme = "ws" if parsed.scheme in {"http", "ws"} else "wss"
    return urlunparse(
        (
            scheme,
            parsed.netloc,
            f"/v1/ingest/audio/{session_id}",
            "",
            "",
            "",
        )
    )


def create_local_audio_ingest_app(
    *,
    manager: Optional[LocalAudioIngestSessionManager] = None,
) -> FastAPI:
    session_manager = manager or LocalAudioIngestSessionManager()
    app = FastAPI(title="Local Audio Ingest Service")

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/v1/ingest/audio/{session_id}")
    async def get_audio_session(session_id: str) -> Dict[str, Any]:
        try:
            return await session_manager.get_session(session_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": f"unknown session: {session_id}", "type": "not_found"}},
            ) from error

    @app.websocket("/v1/ingest/audio/{session_id}")
    async def publish_audio(
        websocket: WebSocket,
        session_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_format: str = "s16le",
        source_label: str = "chrome-extension",
    ) -> None:
        await websocket.accept()
        source = str(websocket.url)
        try:
            snapshot = await session_manager.open_publisher(
                session_id=session_id,
                source=source,
                source_label=source_label,
                sample_rate=sample_rate,
                channels=channels,
                sample_format=sample_format,
            )
            await websocket.send_text(
                json.dumps({"event_type": "source.connected", "session": snapshot})
            )
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload:
                    await session_manager.publish_audio(session_id=session_id, frame=payload)
                    continue
                text_payload = message.get("text")
                if text_payload:
                    try:
                        control = json.loads(text_payload)
                    except json.JSONDecodeError:
                        control = {}
                    if control.get("type") == "finish":
                        break
        except WebSocketDisconnect:
            pass
        except ValueError as error:
            await websocket.send_text(
                json.dumps(
                    {
                        "event_type": "source.error",
                        "session_id": session_id,
                        "error": str(error),
                    }
                )
            )
        finally:
            try:
                await session_manager.close_publisher(session_id=session_id)
            except ValueError:
                pass
            try:
                await websocket.close()
            except RuntimeError:
                pass

    @app.websocket("/v1/ingest/audio/{session_id}/consume")
    async def consume_audio(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        consumer_id, queue, snapshot = await session_manager.open_consumer(
            session_id=session_id,
            source=str(websocket.url),
        )
        try:
            await websocket.send_text(
                json.dumps({"event_type": "source.snapshot", "session": snapshot})
            )
            while True:
                item = await queue.get()
                if item is LOCAL_AUDIO_FRAME_SENTINEL:
                    await websocket.send_text(
                        json.dumps({"event_type": "source.closed", "session_id": session_id})
                    )
                    break
                await websocket.send_bytes(item)
        except WebSocketDisconnect:
            pass
        finally:
            await session_manager.close_consumer(session_id=session_id, consumer_id=consumer_id)
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app


app = create_local_audio_ingest_app()
