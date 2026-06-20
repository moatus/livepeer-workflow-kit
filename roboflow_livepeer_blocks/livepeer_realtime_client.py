"""Remote Livepeer ws-realtime client for true-streaming transcription."""

from __future__ import annotations

import json
import math
import os
import ssl
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode, urlparse, urlunparse

from .config import (
    DEFAULT_OPEN_CLEARINGHOUSE_URL,
    DEFAULT_TRUE_STREAMING_CAPABILITY,
    DEFAULT_TRUE_STREAMING_OFFERING,
)
from .true_streaming import iter_pcm16_wav_frames

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only in minimal environments
    httpx = None  # type: ignore[assignment]


LIVEPEER_REALTIME_SDK_IDENTITY = "roboflow-livepeer-blocks-realtime/0.1.0"


def _payment_header_value(payment_envelope: Any) -> str:
    if isinstance(payment_envelope, (dict, list)):
        return json.dumps(payment_envelope)
    return str(payment_envelope or "")


def _redact_handshake_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() == "livepeer-payment":
            redacted[key] = "[redacted]" if value else ""
        else:
            redacted[key] = value
    return redacted


def _handshake_request_diagnostics(headers: Dict[str, str]) -> Dict[str, Any]:
    return {
        "header_value_lengths": {key: len(value) for key, value in headers.items()},
        "header_value_has_newline": {
            key: ("\n" in value or "\r" in value) for key, value in headers.items()
        },
    }


def _websocket_runtime_diagnostics(connect: Any) -> Dict[str, Any]:
    proxy_env_names = sorted(
        key
        for key in os.environ
        if "PROXY" in key.upper()
        or key
        in {
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
            "PYTHONHTTPSVERIFY",
        }
    )
    details: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "openssl_version": ssl.OPENSSL_VERSION,
        "proxy_env_names": proxy_env_names,
        "websocket_connect_module": getattr(connect, "__module__", ""),
        "websocket_connect_name": getattr(connect, "__name__", type(connect).__name__),
    }
    try:
        import websockets

        details["websockets_version"] = str(getattr(websockets, "__version__", ""))
    except Exception:
        details["websockets_version"] = ""
    return details


def _header_items(headers: Any) -> Iterable[tuple[str, str]]:
    if headers is None:
        return []
    if hasattr(headers, "raw_items"):
        try:
            return [(str(key), str(value)) for key, value in headers.raw_items()]
        except Exception:
            return []
    if hasattr(headers, "items"):
        try:
            return [(str(key), str(value)) for key, value in headers.items()]
        except Exception:
            return []
    return []


def _handshake_response_diagnostics(error: BaseException) -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    response = getattr(error, "response", None)
    status = None
    for source in (response, error):
        if source is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(source, attr, None)
            if value is not None:
                status = value
                break
        if status is not None:
            break
    if status is not None:
        try:
            details["broker_http_status"] = int(status)
        except (TypeError, ValueError):
            details["broker_http_status"] = str(status)

    response_headers: Dict[str, str] = {}
    for source in (response, error):
        if source is None:
            continue
        for key, value in _header_items(getattr(source, "headers", None)):
            if key.lower().startswith("livepeer-"):
                response_headers[key] = value
    if response_headers:
        details["broker_response_headers"] = response_headers
    return details


def _handshake_failure_event(
    *,
    broker_url: str,
    websocket_url: str,
    headers: Dict[str, str],
    error: BaseException,
) -> Dict[str, Any]:
    event = {
        "event_type": "livepeer.realtime.websocket.handshake.failed",
        "broker_url": broker_url,
        "websocket_url": websocket_url,
        "headers": _redact_handshake_headers(headers),
        "header_names": sorted(headers.keys()),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    event.update(_handshake_request_diagnostics(headers))
    event.update(_handshake_response_diagnostics(error))
    return event


def _handshake_runtime_error(event: Dict[str, Any]) -> RuntimeError:
    parts = ["Livepeer realtime websocket handshake failed"]
    if "broker_http_status" in event:
        parts.append(f"broker_http_status={event['broker_http_status']}")
    for key, value in sorted(event.get("broker_response_headers", {}).items()):
        parts.append(f"{key}={value}")
    if event.get("error"):
        parts.append(f"error={event['error']}")
    return RuntimeError("; ".join(parts))


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


def _coerce_int(value: Any, *, default: Optional[int] = None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        if default is not None:
            return default
        raise RuntimeError(f"Invalid integer value: {value!r}")


def _append_path(base_url: str, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"{base_url.rstrip('/')}/{path_or_url.lstrip('/')}"


def _work_units_from_events(
    *,
    events: List[Dict[str, Any]],
    sample_rate: int,
    sent_audio_bytes: int,
    work_unit: str,
) -> int:
    if sent_audio_bytes <= 0:
        return 0
    if work_unit not in {"seconds", "audio_seconds"}:
        return max(1, math.ceil(sent_audio_bytes / 2 / sample_rate))
    observed_duration = 0.0
    for event in events:
        try:
            observed_duration = max(observed_duration, float(event.get("duration_seconds") or 0.0))
        except (TypeError, ValueError):
            continue
    if observed_duration <= 0:
        observed_duration = sent_audio_bytes / 2 / sample_rate
    return max(1, math.ceil(observed_duration))


class LivepeerRemoteTrueStreamingWebSocketClient:
    """Client for Livepeer's ws-realtime true-streaming transcription capability."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        base_url: str = DEFAULT_OPEN_CLEARINGHOUSE_URL,
        capability: str = DEFAULT_TRUE_STREAMING_CAPABILITY,
        offering: str = DEFAULT_TRUE_STREAMING_OFFERING,
        estimated_runway_units: Optional[int] = None,
        max_total_units: Optional[int] = None,
        websocket_connect: Any = None,
        http_client: Any = None,
        receive_timeout_seconds: float = 0.05,
        initial_receive_timeout_seconds: Optional[float] = None,
        finish_receive_timeout_seconds: float = 5.0,
    ) -> None:
        if not api_key:
            raise ValueError("LIVEPEER_OPEN_CLEARINGHOUSE_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.capability = capability
        self.offering = offering
        self.estimated_runway_units = estimated_runway_units
        self.max_total_units = max_total_units
        self._websocket_connect = websocket_connect
        self.receive_timeout_seconds = receive_timeout_seconds
        self.initial_receive_timeout_seconds = (
            initial_receive_timeout_seconds if initial_receive_timeout_seconds is not None else 30.0
        )
        self.finish_receive_timeout_seconds = finish_receive_timeout_seconds
        if http_client is not None:
            self._client = http_client
        else:
            if httpx is None:
                raise RuntimeError("httpx is required to call Livepeer realtime sessions")
            self._client = httpx.Client(timeout=120.0)

    def connect_session(
        self,
        *,
        session_id: str,
        language: str = "en",
        preset: str = "meeting",
        max_speakers: int = 4,
        sample_rate: int = 16000,
        frame_duration_seconds: float = 0.08,
    ) -> "_LivepeerRemoteTrueStreamingWebSocketSession":
        offering = self._discover_offering()
        interaction_mode = str(offering.get("extra", {}).get("interaction_mode") or "")
        if interaction_mode and interaction_mode != "ws-realtime@v0":
            raise RuntimeError(
                f"Livepeer offering {self.offering!r} uses {interaction_mode!r}, not ws-realtime@v0"
            )
        remote_sample_rate = offering.get("extra", {}).get("streaming", {}).get("sample_rate")
        if remote_sample_rate and int(remote_sample_rate) != int(sample_rate):
            raise RuntimeError(
                f"Livepeer offering {self.offering!r} requires sample_rate={remote_sample_rate}, got {sample_rate}"
            )
        estimated_runway_units = self.estimated_runway_units or 60
        max_total_units = self.max_total_units or max(estimated_runway_units + 30, estimated_runway_units)
        open_response = self._client.post(
            f"{self.base_url}/v1/sessions",
            headers={
                "X-API-Key": self.api_key,
                "Livepeer-Open-Clearinghouse-SDK": LIVEPEER_REALTIME_SDK_IDENTITY,
            },
            json={
                "capability": self.capability,
                "offering": self.offering,
                "estimated_runway_units": estimated_runway_units,
                "max_total_units": max_total_units,
            },
        )
        open_response.raise_for_status()
        session_info = open_response.json()
        return _LivepeerRemoteTrueStreamingWebSocketSession(
            base_url=self.base_url,
            api_key=self.api_key,
            capability=self.capability,
            offering=self.offering,
            work_unit=str(offering.get("work_unit") or offering.get("extra", {}).get("work_unit") or "seconds"),
            session_request_id=session_id,
            language=language,
            preset=preset,
            max_speakers=max_speakers,
            sample_rate=sample_rate,
            frame_duration_seconds=frame_duration_seconds,
            session_info=session_info,
            http_client=self._client,
            websocket_connect=self._websocket_connect,
            receive_timeout_seconds=self.receive_timeout_seconds,
            initial_receive_timeout_seconds=self.initial_receive_timeout_seconds,
            finish_receive_timeout_seconds=self.finish_receive_timeout_seconds,
        )

    def _discover_offering(self) -> Dict[str, Any]:
        response = self._client.get(
            f"{self.base_url}/v1/capabilities",
            headers={
                "X-API-Key": self.api_key,
                "Livepeer-Open-Clearinghouse-SDK": LIVEPEER_REALTIME_SDK_IDENTITY,
            },
        )
        response.raise_for_status()
        payload = response.json()
        for capability in payload.get("items", []):
            if capability.get("name") != self.capability:
                continue
            for offering in capability.get("offerings", []):
                if offering.get("id") == self.offering:
                    merged = dict(offering)
                    merged.setdefault("work_unit", capability.get("work_unit"))
                    return merged
        raise RuntimeError(
            f"Livepeer capability/offering not found: {self.capability!r}/{self.offering!r}"
        )


class _LivepeerRemoteTrueStreamingWebSocketSession:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        capability: str,
        offering: str,
        work_unit: str,
        session_request_id: str,
        language: str,
        preset: str,
        max_speakers: int,
        sample_rate: int,
        frame_duration_seconds: float,
        session_info: Dict[str, Any],
        http_client: Any,
        websocket_connect: Any,
        receive_timeout_seconds: float,
        initial_receive_timeout_seconds: float,
        finish_receive_timeout_seconds: float,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.capability = capability
        self.offering = offering
        self.work_unit = work_unit
        self.session_request_id = session_request_id
        self.billing_session_id = str(session_info.get("session_id") or session_request_id)
        self.language = language
        self.preset = preset
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate
        self.frame_duration_seconds = frame_duration_seconds
        self.session_info = session_info
        self.receive_timeout_seconds = receive_timeout_seconds
        self.initial_receive_timeout_seconds = initial_receive_timeout_seconds
        self.finish_receive_timeout_seconds = finish_receive_timeout_seconds
        self.events: List[Dict[str, Any]] = [
            {
                "event_type": "payment.session.opened",
                "billing_session_id": session_info.get("session_id", ""),
                "work_id": session_info.get("work_id", ""),
                "broker_url": session_info.get("broker_url", ""),
                "mode": session_info.get("mode", ""),
                "capability": capability,
                "offering": offering,
                "expected_value_wei": session_info.get("expected_value_wei", 0),
                "funded_value_wei": session_info.get("funded_value_wei", 0),
            }
        ]
        self._client = http_client
        self._connect = websocket_connect or _load_websocket_connect()
        self._websocket: Any = None
        self._websocket_context: Any = None
        self._closed = False
        self._finished = False
        self._sent_audio_bytes = 0

    def __enter__(self) -> "_LivepeerRemoteTrueStreamingWebSocketSession":
        ws_url = self._ws_url()
        headers = self._headers()
        broker_url = str(self.session_info.get("broker_url") or "")
        self.events.append(
            {
                "event_type": "livepeer.realtime.websocket.handshake.prepared",
                "broker_url": broker_url,
                "websocket_url": ws_url,
                "requested_session_id": self.session_request_id,
                "billing_session_id": self.billing_session_id,
                "headers": _redact_handshake_headers(headers),
                "header_names": sorted(headers.keys()),
                "runtime": _websocket_runtime_diagnostics(self._connect),
                **_handshake_request_diagnostics(headers),
            }
        )
        try:
            self._websocket_context = self._connect(
                ws_url,
                additional_headers=headers,
                open_timeout=self.initial_receive_timeout_seconds,
                close_timeout=5.0,
            )
        except BaseException as error:
            failure_event = _handshake_failure_event(
                broker_url=broker_url,
                websocket_url=ws_url,
                headers=headers,
                error=error,
            )
            self.events.append(failure_event)
            if isinstance(error, Exception):
                raise _handshake_runtime_error(failure_event) from error
            raise
        self._websocket = self._websocket_context
        if hasattr(self._websocket_context, "__enter__"):
            try:
                self._websocket = self._websocket_context.__enter__()
            except BaseException as error:
                failure_event = _handshake_failure_event(
                    broker_url=broker_url,
                    websocket_url=ws_url,
                    headers=headers,
                    error=error,
                )
                self.events.append(failure_event)
                if isinstance(error, Exception):
                    raise _handshake_runtime_error(failure_event) from error
                raise
        return self

    def __exit__(self, *_: Any) -> None:
        try:
            if not self._finished:
                self.finish()
        finally:
            self._close_ws()

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
            self._sent_audio_bytes += len(frame)
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
        self._sent_audio_bytes += len(frame)
        self._websocket.send(frame)
        self._drain_received_events()
        emitted = self.events[start_index:]
        if source_event:
            for event in emitted:
                event.setdefault("source_event", source_event)
        return emitted

    def finish(self) -> List[Dict[str, Any]]:
        if self._finished:
            return []
        self._finished = True
        start_index = len(self.events)
        if self._websocket is not None:
            self._websocket.send(json.dumps({"type": "finish"}))
            self._drain_received_events(
                until_finished=True,
                receive_timeout_seconds=self.finish_receive_timeout_seconds,
            )
        actual_units = _work_units_from_events(
            events=self.events,
            sample_rate=self.sample_rate,
            sent_audio_bytes=self._sent_audio_bytes,
            work_unit=self.work_unit,
        )
        outcome = "ok"
        try:
            close_response = self._client.post(
                _append_path(self.base_url, str(self.session_info.get("close_endpoint") or "")),
                headers={
                    "X-API-Key": self.api_key,
                    "Livepeer-Open-Clearinghouse-SDK": LIVEPEER_REALTIME_SDK_IDENTITY,
                },
                json={
                    "actual_units": actual_units,
                    "outcome": outcome,
                },
            )
            close_response.raise_for_status()
            self.events.append(
                {
                    "event_type": "payment.session.closed",
                    "billing_session_id": self.session_info.get("session_id", ""),
                    "work_id": self.session_info.get("work_id", ""),
                    "actual_units": actual_units,
                    "settlement": close_response.json(),
                }
            )
        finally:
            self._close_ws()
            self._closed = True
        return self.events[start_index:]

    def _ws_url(self) -> str:
        broker_url = str(self.session_info.get("broker_url") or "").rstrip("/")
        parsed = urlparse(broker_url)
        scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
        audio_query = urlencode(
            {
                "session_id": self.billing_session_id,
                "language": self.language,
                "preset": self.preset,
                "max_speakers": self.max_speakers,
                "sample_rate": self.sample_rate,
            }
        )
        direct_path = parsed.path or "/"
        direct_query = f"{parsed.query}&{audio_query}" if parsed.query else audio_query
        return urlunparse((scheme, parsed.netloc, direct_path, "", direct_query, ""))

    def _headers(self) -> Dict[str, str]:
        return {
            "Livepeer-Capability": self.capability,
            "Livepeer-Offering": self.offering,
            "Livepeer-Mode": str(self.session_info.get("mode") or ""),
            "Livepeer-Spec-Version": "0.1",
            "Livepeer-Request-Id": str(uuid.uuid4()),
            "Livepeer-Payment": _payment_header_value(self.session_info.get("payment_envelope")),
        }

    def _drain_received_events(
        self,
        *,
        minimum: int = 0,
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
                if received >= minimum:
                    return
                raise
            except Exception:
                if until_finished or received >= minimum:
                    return
                raise
            if raw is None:
                if until_finished and saw_finished:
                    return
                if received >= minimum:
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
            if not until_finished and minimum > 0 and received >= minimum:
                return

    def _close_ws(self) -> None:
        if self._websocket_context is not None and hasattr(self._websocket_context, "__exit__"):
            try:
                self._websocket_context.__exit__(None, None, None)
            except Exception:
                return


def _load_websocket_connect() -> Any:
    try:
        from websockets.sync.client import connect
    except ImportError as error:  # pragma: no cover - exercised in minimal environments
        raise RuntimeError(
            "websockets is required for Livepeer remote true-streaming sessions"
        ) from error
    return connect
